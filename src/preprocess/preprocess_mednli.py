# src/preprocess/preprocess_mednli.py
import pandas as pd
import json
import os
import argparse

def preprocess_mednli(input_train, input_val, input_test, output_dir, sample_fraction=1.0):
    """Preprocess MedNLI dataset with optional sampling."""
    def read_jsonl(file_path):
        with open(file_path) as f:
            return [json.loads(line) for line in f]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and sample data
    train = read_jsonl(input_train)
    val = read_jsonl(input_val)
    test = read_jsonl(input_test)
    
    if sample_fraction < 1.0:
        import random
        random.seed(42)
        train = random.sample(train, int(len(train) * sample_fraction))
        val = random.sample(val, int(len(val) * sample_fraction))
        test = random.sample(test, int(len(test) * sample_fraction))
    
    # Save as JSONL files (keeping original format)
    def save_jsonl(data, filepath):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val, f"{output_dir}/val.jsonl")
    save_jsonl(test, f"{output_dir}/test.jsonl")
    
    print(f"Saved processed data to {output_dir} with:")
    print(f"- Train: {len(train)} examples")
    print(f"- Val: {len(val)} examples")
    print(f"- Test: {len(test)} examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess MedNLI dataset')
    parser.add_argument('--input_train', required=True, help='Path to train JSONL file')
    parser.add_argument('--input_val', required=True, help='Path to validation JSONL file')
    parser.add_argument('--input_test', required=True, help='Path to test JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                      help='Fraction of data to use (default: 1.0)')
    
    args = parser.parse_args()
    preprocess_mednli(
        args.input_train,
        args.input_val,
        args.input_test,
        args.output_dir,
        args.sample_fraction
    )