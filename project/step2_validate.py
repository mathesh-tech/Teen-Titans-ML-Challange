import pandas as pd
import os
import sys

def main():
    file_path = "output/candidate_pairs.csv"
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    print("Validating candidate pairs...")
    
    total_pairs = 0
    unique_s1 = set()
    unique_target = set()
    
    # Process in chunks to prevent 16GB RAM crash
    chunksize = 1000000
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=str):
        total_pairs += len(chunk)
        unique_s1.update(chunk.iloc[:, 0].unique())
        unique_target.update(chunk.iloc[:, 1].unique())
        
    avg_cands = total_pairs / len(unique_s1) if len(unique_s1) > 0 else 0
    
    report = (
        f"Total candidate pairs: {total_pairs:,}\n"
        f"Unique source1 ids: {len(unique_s1):,}\n"
        f"Unique target ids: {len(unique_target):,}\n"
        f"Average candidates per source1 entity: {avg_cands:.2f}\n"
    )
    
    with open("output/candidate_stats.txt", "w") as f:
        f.write(report)
        
    print(report)
    
    if total_pairs > 5000000:
        print("CANDIDATE EXPLOSION DETECTED")
        print("Deleting corrupted 3.1 GB file to prevent OOM loop...")
        os.remove(file_path)

if __name__ == "__main__":
    main()
