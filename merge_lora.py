"""
merge_adapter.py

Merge the trained LoRA adapter back into the base Llama model
and save the full merged model for inference.

Usage:
    python merge_adapter.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_dir ./models/llama-3.1-8b-therapist-french \
        --output_dir ./models/llama-3.1-8b-therapist-french-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()