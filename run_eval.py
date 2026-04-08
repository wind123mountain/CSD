import argparse
from evaluator import Evaluator

import torch
import json
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--student_device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--data_dir", type=str, default='processed_data/ace/qwen/')
    parser.add_argument("--dataset_name", type=str, default='ace')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.lora_path is not None:
        evaluator = Evaluator(
            tokenizer_path=args.tokenizer,
            model_type=args.model_type,
            model_path=args.model_path,
            distilled_lora=args.lora_path,
            device=args.student_device,
            seeds=[42]
        )
    else:
        evaluator = Evaluator(
            tokenizer_path=args.tokenizer,
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.student_device,
            seeds=[42]
        )
    
    evaluator.model.config.output_hidden_states=False
    evaluator.model.config.output_attentions=False

    
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    with torch.cuda.amp.autocast(dtype=dtype):
        metrics, responses = evaluator.evaluate_benchmark_dataset(
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,            
            batch_size=args.val_batch_size, 
            max_length=1024, max_prompt_length=512, split="test"
        )

    with open(args.output_dir + f"/{args.dataset_name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    with open(args.output_dir + f"/{args.dataset_name}_answers.jsonl", "w") as f:
        for resp in responses:
            f.write(json.dumps({"text": resp}) + "\n")

if __name__ == "__main__":
    main()