import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
# from ed_eval import ed_evaluate
from rouge_metric import compute_metrics
from peft import PeftModel
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
import json
from transformers import set_seed
import re
from data_utils.lm_datasets import LMEvalDataset
import torch.nn.functional as F
import random

class Args:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.max_length = 1024
        self.max_prompt_length = 512

        for k, v in kwargs.items():
            setattr(self, k, v)

class Evaluator: 
    def __init__(self, tokenizer_path: str, model_type,
                 model_path: str | None = None,
                 sft_lora: str | None = None,
                 distilled_lora: str | None = None,
                 device: str = 'cuda', seeds: list[int] = [10,20,30,40,50]):
        self.device = device
        self.args = Args(model_type=model_type)

        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            if sft_lora is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    sft_lora
                ).merge_and_unload()
            if distilled_lora is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    distilled_lora
                ).merge_and_unload()

            self.model.to(device)
        else:
            self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seeds = seeds
    
    def evaluate(self, dataset: LMEvalDataset, batch_size, max_length):
        collate_fn = dataset.collate

        generation_config = GenerationConfig(
            do_sample=True,
            top_p=0.95,
            temperature=0.7,            
            max_length=max_length,
            min_length=None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        self.model.eval()
        
        all_response_ids = []
        
        with torch.no_grad():
            for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating")):
                dataset.move_to_device(model_batch, no_model_batch, gen_data, self.device)
                
                max_new_tokens = max_length - gen_data["input_ids"].size(1)
                       
                gen_out = self.model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens
                )
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, max_length - full_ids.shape[1]),
                    value=self.tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                        
                       
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)

        references = dataset.answers
        responses = responses[:len(references)]
        
        results = compute_metrics(responses, references)

        # ed_metrics = ed_evaluate(responses, references)
        # results.update(ed_metrics)

        return results, responses

    @torch.no_grad()
    def evaluate_benchmark_dataset(
        self, 
        data_dir: str, 
        dataset_name: str,
        batch_size: int = 10,
        max_length: int = 1024,
        max_prompt_length: int = 512,
        split: str = "test",
    ):
        set_seed(self.seeds[0])

        self.args.max_length = max_length
        self.args.max_prompt_length = max_prompt_length
        
        rng_sample = random.Random(self.seeds[0])
        test_dataset = LMEvalDataset(self.args, self.tokenizer, data_dir, split, rng_sample)

        metrics, responses = self.evaluate(test_dataset, batch_size, max_length)
        print(dataset_name, ": ", metrics)

        return metrics, responses
    
    @torch.no_grad()
    def evaluate_multiple_benchmarks(
        self,
        benchmark_configs: Dict[str, str],
        batch_size: int = 10,
        max_seq_length: int = 256,
        max_new_tokens: int = 384
    ) -> Dict[str, Dict]:
        """
        Evaluate model on multiple benchmark datasets
        
        Args:
            benchmark_configs: Dictionary mapping dataset keys to file paths
                Example: {
                    "dolly": "/path/to/dolly/valid.jsonl",
                    "self_instruct": "/path/to/self_instruct/valid.jsonl"
                }
            batch_size: Batch size for evaluation
            max_seq_length: Maximum input sequence length
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary with results for each benchmark
        """
        results = {}
        
        # Dataset name mapping
        dataset_names = {
            "dolly": "Dolly",
            "self_instruct": "Self-Instruct", 
            "vicuna": "Vicuna",
            "sni": "S-NI",
            "unni": "UnNI"
        }
        
        for key, dataset_path in benchmark_configs.items():
            dataset_name = dataset_names.get(key, key.title())
            
            if dataset_path and os.path.exists(dataset_path):
                try:
                    score = self.evaluate_benchmark_dataset(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length,
                        max_new_tokens=max_new_tokens
                    )
                    results[key] = {
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "rouge_l_f1": score,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"Error evaluating {dataset_name}: {str(e)}")
                    results[key] = {
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "rouge_l_f1": None,
                        "status": "error",
                        "error_message": str(e)
                    }
            else:
                print(f"Warning: Dataset path for {dataset_name} not found: {dataset_path}")
                results[key] = {
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "rouge_l_f1": None,
                    "status": "not_found"
                }
        
        return results
    
    def evaluate_gsm8k(
        self,
        batch_size: int = 64,
        max_new_tokens: int = 512,
        system_prompt: str = "Put your final answer within \\boxed{}.",
        seed: int = 42,
    ) -> dict:
        set_seed(seed)
        dataset = load_dataset("gsm8k", "main", split="test")
        total = len(dataset)
        correct = 0
        gt_list, pred_list = [], []

        progress = tqdm(range(0, total, batch_size), desc="GSM8K", unit="batch")

        for i in progress:
            batch = dataset[i:i + batch_size]
            prompts, gt_answers = [], []

            for question, answer in zip(batch['question'], batch['answer']):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                gt_answers.append(extract_gsm8k_answer(answer))

            gt_list.extend(gt_answers)

            inputs = self.tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred, gt in zip(decoded, gt_answers):
                pred_ans = extract_math_answer(pred)
                if pred_ans is not None and gt is not None and pred_ans == gt:
                    correct += 1
                pred_list.append(pred_ans)

            # ── DEBUG: chỉ print batch đầu tiên ──
            if i == 0:
                print("\n" + "="*60)
                print(f"PROMPT MẪU:\n{repr(prompts[0][-300:])}")
                print(f"\nOUTPUT MẪU (raw):\n{decoded[0][-400:]}")
                print(f"\nGT answer: {gt_answers[0]}")
                print(f"Extracted pred: {pred_list[0]}")
                print("="*60 + "\n")
                
            if i % (batch_size * 5) == 0:
                print(f"[{i}/{total}] Acc so far: {correct / (i + len(batch['question'])):.4f}")

        pass_at_1 = correct / total
        print(f"\nPASS@1 on GSM8K: {pass_at_1:.4%}")
        return {"pass@1": round(pass_at_1 * 100, 4), "correct": correct,
                "total": total, "predictions": pred_list, "references": gt_list}

    @torch.no_grad()
    def evaluate_math500(
        self,
        batch_size: int = 32,
        max_new_tokens: int = 512,
        system_prompt: str = "Put your final answer within \\boxed{}.",
        seed: int = 42,
    ) -> dict:
        set_seed(seed)
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        total = len(dataset)
        correct = 0
        pred_list, gt_list = [], []

        progress = tqdm(range(0, total, batch_size), desc="MATH-500", unit="batch")

        for i in progress:
            batch = dataset[i:i + batch_size]
            prompts, gt_answers = [], []

            for question, answer in zip(batch['problem'], batch['solution']):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                gt_answers.append(extract_math_answer(answer))

            inputs = self.tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred, gt in zip(decoded, gt_answers):
                pred_ans = extract_math_answer(pred)
                if pred_ans is not None and gt is not None and pred_ans == gt:
                    correct += 1
                pred_list.append(pred_ans)
                gt_list.append(gt)

            if i % (batch_size * 10) == 0:
                print(f"[{i}/{total}] Acc so far: {correct / (i + len(batch['problem'])):.4f}")

        pass_at_1 = correct / total
        print(f"\nPASS@1 (MATH-500): {pass_at_1:.4%}")
        return {"pass@1": round(pass_at_1 * 100, 4), "correct": correct,
                "total": total, "predictions": pred_list, "references": gt_list}

    @torch.no_grad()
    def generate_and_save_outputs(
        self,
        dataset_path: str,
        output_file: str,
        batch_size: int = 10,
        max_seq_length: int = 256,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0
    ):
        print(f"\nGenerating outputs for {dataset_path}...")
        
        # Load dataset
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path)['train']
        else:
            dataset = load_dataset(dataset_path, split="train")


        # Preprocess
        processed_dataset = dataset.map(
            lambda x: preprocess_test(x, self.tokenizer, max_seq_length),
            batched=True,
            batch_size=batch_size
        )
    
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "prompt"]
        )
    
        dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)
    
        self.model.eval()
        generations = []
        # set_seed(42)
        set_seed(30)
    
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompts = batch["prompt"]
    
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
    
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for p, gen in zip(prompts, decoded):
                if gen.startswith(p):
                    gen = gen[len(p):].strip()
                generations.append({"prompt": p, "generated_text": gen})

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in generations:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        print(f"Saved {len(generations)} generations to {output_file}")

def extract_gsm8k_answer(text: str) -> str:
    return text.split('####')[-1].strip()


def extract_math_answer(text: str):
    m = re.findall(r'\\boxed\{([^}]*)\}', text)
    if m:
        return m[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return nums[-1] if nums else None



