import os
import sys
import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_generator import generate_dataset, split_dataset

def main():
    """
    Generates synthetic transaction dataset, splits it into train/test sets,
    and saves the results to the data/ directory.
    """
    print("Generating dataset (this may take a few seconds)...")
    df = generate_dataset(n_transactions=10000, fraud_rate=0.05)
    
    print("Splitting dataset...")
    train_df, test_df = split_dataset(df)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving files...")
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    df.head(100).to_csv(os.path.join(data_dir, 'sample_transactions.csv'), index=False)
    
    print("\nSummary Statistics:")
    print(f"Total rows generated: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
    print("\nFraud Type Breakdown:")
    print(df[df['is_fraud'] == 1]['fraud_type'].value_counts())
    print(f"\nFiles saved to {data_dir}/")

if __name__ == "__main__":
    main()
