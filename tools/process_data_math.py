import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import re



def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--max-prompt-length", type=int, default=256)
    group.add_argument("--max_length", type=int, default=512)
    
    group.add_argument("--model-path", type=str)
    group.add_argument("--model-type", type=str)
    group.add_argument("--only-prompt", action="store_true")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_data_args(parser)
    
    args, unknown = parser.parse_known_args()
    return args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, padding_side="right")

    def encode(self, sample):
        arr_str = sample["response"].split("The answer is:")
        arr_str[-1] = " \\boxed{"+ arr_str[-1].strip() +"}"
        reasoning = "The answer is:".join(arr_str[:-1])
        answer = "The answer is: " + arr_str[-1]
        response = reasoning + answer
                
        prompt = Encoder.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Put your final answer within \\boxed{}."},
             {"role": "user", "content": sample["query"]}],
            add_generation_prompt=True,
            tokenize=False
        )
        if re.search(r'<think>', prompt):
            prompt = prompt + '</think>'
            
        full = prompt + '\n' +  response
        
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(full, add_special_tokens=False)
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
        
        return prompt, prompt_tokens, response_tokens, sample["query"], response



def main():
    print("OK")
    args = get_args()
        
    if 'generated' not in args.processed_data_dir:
        args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    dataset = load_dataset('VoCuc/MetaMathQA-50k-256', split='train').shuffle(seed=42)
    # dataset = dataset.shuffle(seed=42).select(range(15000))
    # dataset = [load_dataset('VoCuc/MetaMathQA-50k-256', split='train')[0]]

    train_data = dataset.select(range(1000, 50000))
    valid_data = dataset.select(range(1000))

    all_data = {
            "valid": valid_data,
            "train": train_data
        }
    
    for split in all_data:
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        if args.model_type!="qwen":
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        else:
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)

        # put tokenized data into binary_builder
        inst_num = 0
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, (prompt_str, prompt, response, query_str, response_str) in enumerate(encoded_docs):
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(json.dumps({
                "prompt": prompt_str,
                "query": query_str,
                "response": response_str,
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                print(f"Processed {lid} documents. {inst_num} instances.", f"({lid/elapsed} docs/s).", file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()