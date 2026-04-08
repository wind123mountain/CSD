import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
from ed_eval import ed_evaluate
from rouge_metric import compute_metrics
from peft import PeftModel
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
import json
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

        ed_metrics = ed_evaluate(responses, references)
        results.update(ed_metrics)

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
                # cắt prompt ra để chỉ giữ phần model sinh
                if gen.startswith(p):
                    gen = gen[len(p):].strip()
                generations.append({"prompt": p, "generated_text": gen})

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in generations:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        print(f"Saved {len(generations)} generations to {output_file}")