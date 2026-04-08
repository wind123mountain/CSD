from torch.nn.functional import pad
from torch.utils.data import Dataset
import torch
import json
from transformers import PreTrainedTokenizer
import os

from dataclasses import dataclass


def longest_common_subsequence(a, b, s_i=0, s_j=0) -> list:
    a = a.numpy()
    b = b.numpy()
    m, n = len(a), len(b)
    
    i = s_i
    j = s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j] == 0:
            j += 1
            continue
            
        if a[i][1] == b[j]:
            result.append(i+1)
            i += 1
            j += 1
        elif a[i][1] < b[j]:
            i += 1
        else:
            j += 1

    if len(result) < 2:
        result = [result[-1] - 1] + result
            
    return result

def get_pooler_tensor(segments_idxs):
    # Tạo chỉ số segment đã pad cho toàn bộ batch
    padded_idx_batch = []
    max_seg, max_len_all = 0, 0
    pad_multiple = 4

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))

        padded = torch.stack([
            pad(x, (0, max_len - len(x)), value=-1)
            for x in seg_idx
        ])  # (num_segments, max_len)

        padded_idx_batch.append(padded)

    # Pad toàn bộ batch về cùng shape (B, max_seg, max_len_all)
    def pad2d(t, h, w):
        return pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=-1)

    # max_seg = int(math.ceil(max_seg / pad_multiple) * pad_multiple)
    padded_idx_batch = torch.stack([
        pad2d(p, max_seg, max_len_all) for p in padded_idx_batch
    ])  # (B, max_seg, max_len_all)

    # Tạo mask và gather từ X
    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)

    return {'safe_idx': safe_idx, 'mask': mask}

def prepare_pooler(offset_mapping, starts, phrases_offsets):
    seg_idxs = []
    for offset, start, phrases_offset in zip(offset_mapping, starts, phrases_offsets):

        seg_idx = []
       
        token_offset_start = [start.item()]

        # longest_common_offset = token_offset_start + longest_common_subsequence(offset, phrases_offset, start) 
        longest_common_offset = longest_common_subsequence(offset, phrases_offset, start) 
        student_max_len = 1

        for i in range(1, len(longest_common_offset)):
            seg_idx.append(torch.arange(longest_common_offset[i - 1], longest_common_offset[i]))
            student_max_len = max(student_max_len, seg_idx[-1].size(0))

        seg_idxs.append((seg_idx, student_max_len))

    return get_pooler_tensor(seg_idxs)


class LLMDataset(Dataset):
    def __init__(self, file_path, split, tokenizer, max_len, model_type, return_offsets_mapping=False):

        self.dataset = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_type = model_type
        self.return_offsets_mapping = return_offsets_mapping

        syntactic_parsing_file = os.path.join(file_path, "syntactic_parsing.jsonl")
        prompt_max_len = max_len // 2

        with open(os.path.join(file_path, f"{split}.jsonl"), "r", encoding="utf-8") as f1, \
             open(syntactic_parsing_file, "r", encoding="utf-8") as f2:
            for line1, line2 in zip(f1, f2):
                data = json.loads(line1)
                syntactic_data = json.loads(line2)
                self.dataset.append(data)

                s_prompt = tokenizer(
                    data['prompt'], 
                    max_length=prompt_max_len,
                    truncation=True, 
                    add_special_tokens=False
                )
                data['prompt'] = tokenizer.decode(s_prompt['input_ids'])
                data['prompt_len'] = len(s_prompt['input_ids'])

                prompt_end = len(data['prompt'])

                phrases_lvl1 = [prompt_end] + [prompt_end + item['end_char'] for item in syntactic_data['phrases_lvl1']]
                phrases_lvl2 = [prompt_end] + [prompt_end + item['end_char'] for item in syntactic_data['phrases_lvl2']]
                data['phrases_lvl1_offset'] = torch.tensor(phrases_lvl1)
                data['phrases_lvl2_offset'] = torch.tensor(phrases_lvl2)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (self.dataset[index]['prompt'], 
                self.dataset[index]['output'], 
                self.dataset[index]['prompt_len'],
                self.dataset[index]['phrases_lvl1_offset'], 
                self.dataset[index]['phrases_lvl2_offset'])

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, batch):
        prompts, fulls, prompt_lengths, phrases_lvl1_offsets, phrases_lvl2_offsets = [], [], [], [], []
        for prompt, output, prompt_length, phrases_lvl1_offset, phrases_lvl2_offset in batch:
            prompts.append(prompt)
            fulls.append(prompt + output)
            prompt_lengths.append(prompt_length)
            phrases_lvl1_offsets.append(phrases_lvl1_offset)
            phrases_lvl2_offsets.append(phrases_lvl2_offset)

        inputs = self.tokenizer(
            fulls,
            truncation=True,
            padding=True,
            max_length=self.max_len - 1,
            return_tensors='pt',
            return_offsets_mapping=self.return_offsets_mapping,
            add_special_tokens=False
        )

        eos_tokens = torch.full((inputs["input_ids"].size(0), 1), self.tokenizer.eos_token_id, dtype=torch.long)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], eos_tokens], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], 
                                              torch.zeros((inputs["attention_mask"].size(0), 1), dtype=torch.long)], dim=1)
        
        labels = inputs["input_ids"][:, 1:].clone().detach()
        labels = torch.cat([labels, torch.full((labels.size(0), 1), -100, dtype=torch.long)], dim=1)

        input_lengths = inputs["attention_mask"].sum(dim=1)
        prompt_lengths = torch.tensor(prompt_lengths)

        inputs.pop('position_ids', None)

        if self.model_type in ["gpt2"]:
            position_ids = torch.zeros(inputs['input_ids'].size(), dtype=torch.long)
            for i in range(input_lengths.size(0)):
                position_ids[i, :input_lengths[i]] = torch.arange(0, input_lengths[i], dtype=torch.long)
            inputs["position_ids"] = position_ids

        for i in range(len(labels)):
            labels[i, :(prompt_lengths[i] -1)] = -100
            labels[i, input_lengths[i]:] = -100

        token_offset_mapping = inputs.pop('offset_mapping', None)
        if token_offset_mapping is not None :
            starts = torch.zeros_like(prompt_lengths)
            pooler_tensor = prepare_pooler(token_offset_mapping, starts, phrases_lvl2_offsets)

            inputs['pooler_safe_idx'] = pooler_tensor['safe_idx']
            inputs['pooler_mask'] = pooler_tensor['mask']

        return inputs, {"label": labels}, None
