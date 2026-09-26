import argparse
import json
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--language", type=str, required=True, choices=["ar", "fr", "de"])
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to prepared HF dataset (e.g. ./processed_dataset/fr)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use QLoRA 4-bit quantization")
    parser.add_argument("--no_4bit", action="store_true", default=False, help="Disable 4-bit, use full bf16")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)  # For distributed training
    return parser.parse_args()


def main():
    args = parse_args()
    use_4bit = args.use_4bit and not args.no_4bit

    print(f"{'='*60}")
    print(f"Fine-tuning meta-llama/Meta-Llama-3.1-8B-Instruct for: {args.language}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"QLoRA 4-bit: {use_4bit}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config (QLoRA)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit else None,  # device_map for QLoRA
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 if available
    )
    model.config.use_cache = False  # Disable KV cache for training
    model.config.pretraining_tp = 1

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config 
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset 
    dataset = load_from_disk(args.dataset_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("dev", None)

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # Formatting function
    def formatting_func(example):
       """Convert the stored JSON messages back to chat-template format."""
       messages = json.loads(example["messages"])
       return tokenizer.apply_chat_template(messages, tokenize=False)
    


    # Training config 
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        max_length=args.max_seq_length,
        packing=True,  # Pack multiple short dialogues into one sequence
        seed=args.seed,
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        overwrite_output_dir=False,  # Don't overwrite — needed for checkpoint resume
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    # Train
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_ckpt = os.path.join(
                args.output_dir,
                sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1],
            )
            print(f"Resuming from checkpoint: {last_ckpt}")

    if last_ckpt is None:
        print("Starting training from scratch...")

    trainer.train(resume_from_checkpoint=last_ckpt)

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save the LoRA adapter separately as well
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    print("Done!")


if __name__ == "__main__":
    main()