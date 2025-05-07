import json
import argparse
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
import evaluate

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Metrics
rouge = evaluate.load("rouge")

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\n★ Evaluation Results ★\n"
              f"Loss: {metrics.get('eval_loss', 'NA'):.4f}\n"
              f"ROUGE-L: {metrics.get('eval_rougeL', 'NA'):.2f}\n"
              f"ROUGE-1: {metrics.get('eval_rouge1', 'NA'):.2f}")
              
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Debug prints
    print("\nSample Prediction:", decoded_preds[0])
    print("Sample Reference:", decoded_labels[0])
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v * 100, 4) for k, v in result.items()}

def preprocess_function(examples, max_length=256):
    inputs = ["generate radiology report: " + findings for findings in examples["text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # Load data
    data_files = {
        "train": f"{args.data_dir}/train.jsonl",
        "validation": f"{args.data_dir}/val.jsonl",
        "test": f"{args.data_dir}/test.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, args.max_seq_length),
        batched=True
    )

    # Training setup
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,  # Correct gradient clipping
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_seq_length,
        fp16=args.fp16,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        report_to="none",
        logging_steps=10
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.add_callback(PrintMetricsCallback())
    
    # Train and save
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best_model")
    results = trainer.evaluate(tokenized_datasets["test"])
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f)