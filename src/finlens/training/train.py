"""
QLoRA fine-tuning for SEC filing extraction.
Runs on cloud GPU (T4, A10, L4, A100).

Usage:
    %run train.py --train-data train_chat.jsonl --val-data val_chat.jsonl
"""

import argparse
import json
import os

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_chat_dataset(path: str) -> Dataset:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--train-data", default="data/train_chat.jsonl")
    parser.add_argument("--val-data", default="data/val_chat.jsonl")
    parser.add_argument("--output-dir", default="models/finlens-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--wandb-project", default="finlens")
    args = parser.parse_args()

    # ── Check GPU ──
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. Run on a cloud GPU, not Mac.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ── W&B init ──
    wandb.init(
        project=args.wandb_project,
        name=f"finlens-r{args.lora_rank}-e{args.epochs}-lr{args.lr}",
        config=vars(args),
    )

    # ── Load tokenizer ──
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model in 4-bit (QLoRA) ──
    print("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig( 
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, # Changed to bfloat16
    bnb_4bit_use_double_quant=True, 
    ) 

    model = AutoModelForCausalLM.from_pretrained( 
    args.model, 
    quantization_config=bnb_config, 
    device_map={"": 0}, 
    torch_dtype=torch.bfloat16, # Changed to bfloat16
    token=os.environ.get("HF_TOKEN"), 
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ──
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "gemma" not in args.model.lower():
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Load datasets ──
    print("\nLoading datasets...")
    train_dataset = load_chat_dataset(args.train_data)
    val_dataset = load_chat_dataset(args.val_data)
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val:   {len(val_dataset)} examples")

    # ── SFT Config (replaces TrainingArguments in newer trl) ──
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        report_to="wandb",
        run_name=f"finlens-r{args.lora_rank}-e{args.epochs}",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_seq_length,
    )

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──
    print("\nStarting training...\n")
    trainer.train()

    # ── Save ──
    print(f"\nSaving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    wandb.finish()
    print("\nDone! LoRA adapter saved.")


if __name__ == "__main__":
    main()