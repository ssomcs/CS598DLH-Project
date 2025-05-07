import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
import argparse

def preprocess_mimic_cxr(input_csv, output_dir, sample_fraction=1.0):
    # Load data
    df = pd.read_csv(input_csv)
    
    # Sample if needed
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
    
    # Split into train/val/test (80/10/10)
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Convert to datasets
    train_ds = Dataset.from_pandas(train)
    val_ds = Dataset.from_pandas(val)
    test_ds = Dataset.from_pandas(test)
    
    # Save as JSONL
    train_ds.to_json(f"{output_dir}/train.jsonl")
    val_ds.to_json(f"{output_dir}/val.jsonl")
    test_ds.to_json(f"{output_dir}/test.jsonl")
    
    print(f"Processed {len(df)} examples, split into:")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-CXR dataset')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
    parser.add_argument('--sample_fraction', type=float, default=1.0, 
                       help='Fraction of data to use (default: 1.0 for full dataset)')
    
    args = parser.parse_args()
    preprocess_mimic_cxr(args.input_csv, args.output_dir, args.sample_fraction)