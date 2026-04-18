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
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--student_device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--data_dir", type=str, default='processed_data/MetaMathQA-50k/qwen/')
    parser.add_argument("--dataset_name", type=str, default='ace')
    # math benchmarks
    parser.add_argument("--gsm8k", action="store_true", help="Run GSM8K evaluation")
    parser.add_argument("--math500", action="store_true", help="Run MATH-500 evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

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

    evaluator.model.config.output_hidden_states = False
    evaluator.model.config.output_attentions = False

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ── GSM8K ─────────────────────────────────────────────────────────────────
    if args.gsm8k:
        with torch.cuda.amp.autocast(dtype=dtype):
            gsm8k_metrics = evaluator.evaluate_gsm8k(
                batch_size=args.val_batch_size,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        if args.output_dir:
            out = {k: v for k, v in gsm8k_metrics.items() if k != "predictions"}
            with open(f"{args.output_dir}/gsm8k_eval.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)

    # ── MATH-500 ──────────────────────────────────────────────────────────────
    if args.math500:
        with torch.cuda.amp.autocast(dtype=dtype):
            math500_metrics = evaluator.evaluate_math500(
                batch_size=args.val_batch_size,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        if args.output_dir:
            out = {k: v for k, v in math500_metrics.items() if k != "predictions"}
            with open(f"{args.output_dir}/math500_eval.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)

    if not (args.gsm8k and args.math500):
        with torch.cuda.amp.autocast(dtype=dtype):
            metrics, responses = evaluator.evaluate_benchmark_dataset(
                data_dir=args.data_dir,
                dataset_name=args.dataset_name,
                batch_size=args.val_batch_size,
                max_length=1024, max_prompt_length=512, split="valid"
            )

        if args.output_dir:
            with open(f"{args.output_dir}/{args.dataset_name}_eval.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            with open(f"{args.output_dir}/{args.dataset_name}_answers.jsonl", "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")

if __name__ == "__main__":
    main()