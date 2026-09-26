import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_cleaner import DataCleaner
from src.candidate_generator import CandidateGenerator
from src.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.WARNING)

def test_50_random():
    print("=" * 85)
    print("          RANDOM 50-ENTITY ML INFERENCE & MATCH TEST")
    print("=" * 85)
    
    model_path = os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")
    threshold_path = os.path.join(BASE_DIR, "output", "best_threshold.json")
    
    model = joblib.load(model_path)
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
    threshold = threshold_data.get("best_threshold", 0.75)
    
    print(f"[OK] Model Loaded Artifact : {model_path}")
    print(f"[OK] Classification Threshold: {threshold}\n")
    
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")

    cleaner = DataCleaner()
    df_s1 = pd.read_csv(s1_path, sep='\t', dtype=str)
    sample_s1 = df_s1.sample(n=50, random_state=np.random.randint(0, 10000)).copy()
    
    print("Sampling 50 random Source 1 entity records...")
    print("Loading target candidate pool (Source 2 + Source 3)...")
    
    df_s2 = pd.read_csv(s2_path, sep='\t', dtype=str)
    df_s3 = pd.read_csv(s3_path, sep='\t', dtype=str)
    
    clean_s1 = cleaner.process(sample_s1)
    clean_s2 = cleaner.process(df_s2)
    clean_s3 = cleaner.process(df_s3)
    
    df_target = pd.concat([clean_s2, clean_s3], ignore_index=True)
    
    gen = CandidateGenerator(chunk_size=10000, max_candidates_per_entity=50)
    cand_out = os.path.join(BASE_DIR, "output", "temp_50_candidates.csv")
    gen.generate(clean_s1, clean_s2, clean_s3, output_path=cand_out)
    
    if not os.path.exists(cand_out) or os.path.getsize(cand_out) == 0:
        print("No candidate pairs generated for this 50-entity sample.")
        return

    engineer = FeatureEngineer(chunk_size=10000)
    feat_out = os.path.join(BASE_DIR, "output", "temp_50_features.csv")
    engineer.generate_features(cand_out, clean_s1, df_target, output_path=feat_out)
    
    features_df = pd.read_csv(feat_out)
    candidates = pd.read_csv(cand_out, dtype=str)
    
    exclude_cols = ['source1_entity_id', 'target_entity_id', 'label']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    X_sample = features_df[feature_cols].astype(float)
    y_probs = model.predict(X_sample)
    
    candidates['match_probability'] = y_probs
    candidates['prediction'] = (y_probs >= threshold).astype(int)
    matches = candidates[candidates['prediction'] == 1].copy()
    
    print("\n" + "=" * 85)
    print("               50 RANDOM TEST ENTITIES MATCH RESULTS")
    print("=" * 85)
    print(f"{'#':>2} {'Source1 ID':<15} {'Business Name':<32} {'Country':<7} {'Match Status & Target IDs'}")
    
    high_match_count = 0
    for idx, (_, row) in enumerate(sample_s1.iterrows(), start=1):
        s1_id = str(row['source1_entity_id'])
        name = str(row.get('business_name', 'N/A'))
        if len(name) > 30:
            name = name[:27] + "..."
        country = str(row.get('country', 'N/A'))
        
        matched_rows = matches[matches['source1_entity_id'] == s1_id]
        if len(matched_rows) > 0:
            high_match_count += 1
            target_info = []
            for _, mrow in matched_rows.iterrows():
                target_info.append(f"{mrow['target_entity_id']} ({float(mrow['match_probability'])*100:.1f}%)")
            match_str = ", ".join(target_info)
        else:
            match_str = "NO MATCH"
            
        print(f"{idx:>2} {s1_id:<15} {name:<32} {country:<7} {match_str}")
        
    print("\n" + "=" * 85)
    print("Summary of 50 Random Entities:")
    print(f"  • Total Entities Tested     : 50")
    print(f"  • Entities with High-Match : {high_match_count} ({high_match_count/50*100:.1f}%)")
    print(f"  • Entities without Match   : {50 - high_match_count} ({(50-high_match_count)/50*100:.1f}%)")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    test_50_random()
