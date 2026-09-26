import pandas as pd
import sys
from pathlib import Path

def explore_source(df: pd.DataFrame, dataset_name: str):
    print("=" * 70)
    print(f"--- Data Exploration for: {dataset_name} ---")
    print("=" * 70)
    
    # 1. & 4. Row counts and Column names
    total_records = len(df)
    print(f"Total records: {total_records}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Missing values
    missing = df.isnull().sum()
    print("\nMissing values:")
    for col, count in missing.items():
        pct = (count / total_records) * 100 if total_records > 0 else 0
        print(f"  {col}: {count} ({pct:.2f}%)")
        
    # 3. Duplicate entity_ids
    if 'entity_id' in df.columns:
        dups = df.duplicated(subset=['entity_id']).sum()
        print(f"\nDuplicate entity_ids: {dups}")
        
    print("=" * 70 + "\n")

def explore_ground_truth(df: pd.DataFrame, dataset_name: str):
    print("=" * 70)
    print(f"--- Data Exploration for: {dataset_name} ---")
    print("=" * 70)
    
    total_records = len(df)
    print(f"Total records: {total_records}")
    print(f"Columns: {list(df.columns)}")
    
    missing = df.isnull().sum()
    print("\nMissing values:")
    for col, count in missing.items():
        pct = (count / total_records) * 100 if total_records > 0 else 0
        print(f"  {col}: {count} ({pct:.2f}%)")
        
    # 5. Analyze matched_entity_ids format
    if 'matched_entity_ids' in df.columns:
        valid_matches = df['matched_entity_ids'].dropna()
        match_counts = valid_matches.apply(lambda x: len(str(x).split(',')))
        print("\nMatched entity formats (number of matches per source1_id):")
        print(match_counts.describe(percentiles=[.25, .5, .75, .90, .99]).to_string())
        
        # 6. Estimate positive match distribution
        total_positive_pairs = match_counts.sum()
        print(f"\nEstimated total positive pairs (if fully loaded): {total_positive_pairs:,}")
        
    print("=" * 70 + "\n")

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import DataLoader
    
    try:
        loader = DataLoader()
        print("Analyzing datasets...\n")
        
        df1 = loader.load_source1()
        explore_source(df1, "train_source1.tsv")
        
        df2 = loader.load_source2()
        explore_source(df2, "train_source2.tsv")
        
        df3 = loader.load_source3()
        explore_source(df3, "train_source3.tsv")
        
        gt = loader.load_ground_truth()
        explore_ground_truth(gt, "train_ground_truth.tsv")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
