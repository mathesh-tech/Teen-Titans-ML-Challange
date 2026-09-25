import pandas as pd
import sys
from pathlib import Path

def explore_dataset(df: pd.DataFrame, dataset_name: str):
    """Prints comprehensive exploration statistics for a given DataFrame."""
    print("=" * 70)
    print(f"--- Data Exploration for: {dataset_name} ---")
    print("=" * 70)
    
    # 1. Total records
    total_records = len(df)
    print(f"1. Total records: {total_records}")
    
    # 2. Unique countries
    if 'country' in df.columns:
        unique_countries = df['country'].dropna().unique()
        print(f"2. Unique countries count: {len(unique_countries)}")
        if len(unique_countries) <= 20:
            print(f"   Country list: {list(unique_countries)}")
    else:
        print("2. Unique countries: 'country' column not found.")
        
    # 3. Missing names
    if 'name' in df.columns:
        missing_names = df['name'].isnull().sum()
        pct = (missing_names / total_records) * 100 if total_records > 0 else 0
        print(f"3. Missing names: {missing_names} ({pct:.2f}%)")
    else:
        print("3. Missing names: 'name' column not found.")
        
    # 4. Missing addresses
    if 'address' in df.columns:
        missing_addresses = df['address'].isnull().sum()
        pct = (missing_addresses / total_records) * 100 if total_records > 0 else 0
        print(f"4. Missing addresses: {missing_addresses} ({pct:.2f}%)")
    else:
        print("4. Missing addresses: 'address' column not found.")
        
    # 5. Business name length distribution
    if 'name' in df.columns:
        name_lengths = df['name'].dropna().astype(str).apply(len)
        print("\n5. Business name length distribution (in characters):")
        if not name_lengths.empty:
            print(name_lengths.describe(percentiles=[.25, .5, .75, .90, .99]).to_string())
        else:
            print("   No valid name data to compute lengths.")
    else:
        print("\n5. Business name length distribution: 'name' column not found.")
        
    # 6. Address length distribution
    if 'address' in df.columns:
        addr_lengths = df['address'].dropna().astype(str).apply(len)
        print("\n6. Address length distribution (in characters):")
        if not addr_lengths.empty:
            print(addr_lengths.describe(percentiles=[.25, .5, .75, .90, .99]).to_string())
        else:
            print("   No valid address data to compute lengths.")
    else:
        print("\n6. Address length distribution: 'address' column not found.")
        
    # 7. Sample records
    print("\n7. Sample records (up to 5):")
    if total_records > 0:
        print(df.sample(min(5, total_records)).to_string(index=False))
    else:
        print("   Dataset is empty.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Ensure the script can import from src directory properly
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import DataLoader
    
    try:
        loader = DataLoader()
        print("Attempting to load and explore datasets...\n")
        
        try:
            df1 = loader.load_source1()
            explore_dataset(df1, "train_source1.tsv")
        except FileNotFoundError:
            print("[Warning] train_source1.tsv not found in the dataset folder, skipping...\n")
            
        try:
            df2 = loader.load_source2()
            explore_dataset(df2, "train_source2.tsv")
        except FileNotFoundError:
            print("[Warning] train_source2.tsv not found in the dataset folder, skipping...\n")

        try:
            df3 = loader.load_source3()
            explore_dataset(df3, "train_source3.tsv")
        except FileNotFoundError:
            print("[Warning] train_source3.tsv not found in the dataset folder, skipping...\n")
            
        try:
            gt = loader.load_ground_truth()
            explore_dataset(gt, "train_ground_truth.tsv")
        except FileNotFoundError:
            print("[Warning] train_ground_truth.tsv not found in the dataset folder, skipping...\n")
            
    except Exception as e:
        print(f"An unexpected error occurred during exploration: {e}")
        sys.exit(1)
