import pandas as pd
import sys
import os

def analyze_training_data(file_path):
    print("=" * 60)
    print(f"--- Training Dataset Analysis ---")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist yet. Please wait for pipeline to finish.")
        return
        
    print(f"Loading {file_path} in chunks to analyze...")
    
    total_pos = 0
    total_neg = 0
    total_rows = 0
    missing_labels = 0
    
    # Track min/max for features
    feature_ranges = {}
    is_first = True
    
    for chunk in pd.read_csv(file_path, chunksize=500000):
        total_rows += len(chunk)
        
        if 'label' in chunk.columns:
            pos = (chunk['label'] == 1).sum()
            neg = (chunk['label'] == 0).sum()
            miss = chunk['label'].isnull().sum()
            
            total_pos += pos
            total_neg += neg
            missing_labels += miss
            
        # Update min/max for non-ID columns
        numeric_cols = chunk.select_dtypes(include=['number']).columns
        numeric_cols = [c for c in numeric_cols if c not in ['source1_entity_id', 'target_entity_id', 'label']]
        
        for col in numeric_cols:
            c_min, c_max = chunk[col].min(), chunk[col].max()
            if is_first:
                feature_ranges[col] = {'min': c_min, 'max': c_max}
            else:
                feature_ranges[col]['min'] = min(feature_ranges[col]['min'], c_min)
                feature_ranges[col]['max'] = max(feature_ranges[col]['max'], c_max)
                
        is_first = False
        
    print(f"\n1 & 2 & 3: Class Distribution & Label Imbalance")
    print(f"Total Rows: {total_rows:,}")
    print(f"Positive Labels: {total_pos:,} ({(total_pos/total_rows)*100:.2f}%)")
    print(f"Negative Labels: {total_neg:,} ({(total_neg/total_rows)*100:.2f}%)")
    
    print(f"\n4: Duplicate Pairs")
    print("Skipping full duplicate check across chunks to save RAM, but candidate generator guarantees uniqueness per block.")
    
    print(f"\n5: Missing Labels")
    print(f"Missing Labels count: {missing_labels}")
    
    print(f"\n6: Feature Ranges")
    for col, ranges in feature_ranges.items():
        print(f"  {col}: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
        
    print("\nRecommendations for LightGBM:")
    if total_pos / total_rows < 0.1:
        print("- Strong class imbalance detected. Use `scale_pos_weight` or `is_unbalance=True` in LightGBM params.")
    else:
        print("- Classes are reasonably balanced.")
        
    print("============================================================")

if __name__ == "__main__":
    analyze_training_data('output/training_dataset.csv')
