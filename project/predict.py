"""
predict.py
-----------
Inference pipeline for the Amazon ML Challenge 2026.
Generates predictions on the test dataset using the trained model and optimal threshold.
"""

import os
import sys
import logging
import json
import joblib
import time
import pandas as pd
from pathlib import Path
from typing import Tuple, Any

try:
    import psutil
except ImportError:
    psutil = None

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaner import DataCleaner
from src.candidate_generator import CandidateGenerator
from src.feature_engineer import FeatureEngineer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/predict.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('predict')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming test data will be placed in dataset/test/
TEST_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'test') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Make sure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_test_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the test TSV files with error handling."""
    data_path = Path(data_dir)
    logger.info(f"Loading test data from {data_path}...")
    
    try:
        df1 = pd.read_csv(data_path / 'test_source1.tsv', sep='\t', dtype=str, encoding='utf-8')
        df2 = pd.read_csv(data_path / 'test_source2.tsv', sep='\t', dtype=str, encoding='utf-8')
        df3 = pd.read_csv(data_path / 'test_source3.tsv', sep='\t', dtype=str, encoding='utf-8')
        
        logger.info(f"Loaded Source 1: {df1.shape}")
        logger.info(f"Loaded Source 2: {df2.shape}")
        logger.info(f"Loaded Source 3: {df3.shape}")
        return df1, df2, df3
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}")
        logger.error("Please ensure test_source1.tsv, test_source2.tsv, and test_source3.tsv are in the dataset/test directory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)

def load_artifacts(model_path: str, threshold_path: str) -> Tuple[Any, float]:
    """Loads the trained model and best threshold JSON."""
    logger.info("Loading trained model and threshold artifacts...")
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            
        best_threshold = threshold_data.get('best_threshold', 0.5)
        logger.info(f"Successfully loaded threshold: {best_threshold} from {threshold_path}")
        
        return model, best_threshold
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        logger.error("Please ensure you have run the training pipeline (train.py / pipeline.py) first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        sys.exit(1)

def main():
    logger.info("=" * 70)
    logger.info("  Amazon ML Challenge 2026 — Inference Pipeline")
    logger.info("=" * 70)

    # 1. Load Artifacts
    model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
    threshold_path = os.path.join(OUTPUT_DIR, 'best_threshold.json')
    model, threshold = load_artifacts(model_path, threshold_path)

    # 2. Load Test Data
    df1, df2, df3 = load_test_data(TEST_DATA_DIR)

    # 3. Clean Data
    logger.info("STEP 1: Cleaning test data...")
    cleaner = DataCleaner()
    df1_clean = cleaner.process(df1)
    df2_clean = cleaner.process(df2)
    df3_clean = cleaner.process(df3)
    
    df_target = pd.concat([df2_clean, df3_clean], ignore_index=True)
    logger.info(f"  Combined target pool: {df_target.shape}")

    # 4. Generate or Resume Candidates
    candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 1000:
        logger.info(f"Auto-Resume: {candidate_path} already exists ({os.path.getsize(candidate_path):,} bytes). Skipping generation.")
    else:
        logger.info("STEP 2: Generating candidate pairs for test data...")
        gen = CandidateGenerator(chunk_size=100000, max_candidates_per_entity=2)
        gen.generate(
            df_src1=df1_clean,
            df_src2=df2_clean,
            df_src3=df3_clean,
            output_path=candidate_path
        )
    
    if not os.path.exists(candidate_path) or os.path.getsize(candidate_path) == 0:
        logger.error("No candidates generated for test data. Cannot proceed with predictions.")
        sys.exit(1)

    # 5. Feature Engineering Setup
    logger.info("STEP 3: Preparing feature engineering & TF-IDF models...")
    engineer = FeatureEngineer(chunk_size=100000)
    if not engineer.is_fitted:
        engineer._fit_tfidf(df1_clean, df_target)

    # 6. Stream candidate chunks and predict directly
    logger.info(f"STEP 4: Computing features and predicting in chunks of {engineer.chunk_size:,}...")
    delim = '\t' if str(candidate_path).endswith('.tsv') else ','
    
    total_candidate_rows = sum(1 for _ in open(candidate_path, encoding='utf-8', errors='ignore')) - 1 if os.path.exists(candidate_path) else 0
    logger.info(f"Total candidate pairs to evaluate: {total_candidate_rows:,}")
    
    model_feature_cols = model.feature_name()
    logger.info(f"Model features ({len(model_feature_cols)}): {model_feature_cols}")
    
    matched_pairs = []
    chunk_idx = 0
    total_processed = 0
    start_time = time.time()
    
    for chunk in pd.read_csv(candidate_path, sep=delim, chunksize=engineer.chunk_size, dtype=str):
        chunk_idx += 1
        
        # Compute features for chunk
        feat_chunk = engineer._compute_features_chunk(chunk, df1_clean, df_target)
        
        # Select exact model feature columns
        X_chunk = feat_chunk[model_feature_cols].astype(float)
        
        # Predict probabilities
        y_prob = model.predict(X_chunk)
        
        # Filter positive matches
        mask = (y_prob >= threshold)
        if mask.any():
            matched_df = pd.DataFrame({
                'source1_entity_id': chunk['source1_entity_id'].values[mask],
                'target_entity_id': chunk['target_entity_id'].values[mask],
                'match_probability': y_prob[mask]
            })
            matched_pairs.append(matched_df)
            
        total_processed += len(chunk)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed * 60 if elapsed > 0 else 0
        eta = (total_candidate_rows - total_processed) / rate if rate > 0 else 0
        
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) if psutil else 0
        total_pos = sum(len(m) for m in matched_pairs)
        if chunk_idx % 5 == 0 or total_processed >= total_candidate_rows:
            logger.info(
                f"[PREDICT] Chunk {chunk_idx} | "
                f"Evaluated: {total_processed:,}/{total_candidate_rows:,} | "
                f"Matches Found: {total_pos:,} | "
                f"Rate: {rate:,.0f} pairs/min | "
                f"ETA: {eta:.1f} min | "
                f"RAM: {mem_mb/1024:.2f} GB"
            )
            
    # Combine positive matches
    if matched_pairs:
        positive_matches = pd.concat(matched_pairs, ignore_index=True)
    else:
        positive_matches = pd.DataFrame(columns=['source1_entity_id', 'target_entity_id', 'match_probability'])
        
    logger.info(f"Total positive matches found: {len(positive_matches):,}")
    
    # 7. Format predictions for submission
    logger.info("STEP 5: Formatting predictions for official submission...")
    if not positive_matches.empty:
        # Deduplicate matches if any
        positive_matches = positive_matches.drop_duplicates(subset=['source1_entity_id', 'target_entity_id'])
        # Sort by match_probability descending so strongest matches appear first
        positive_matches = positive_matches.sort_values(by=['source1_entity_id', 'match_probability'], ascending=[True, False])
        submission = positive_matches.groupby('source1_entity_id')['target_entity_id'].apply(
            lambda x: ','.join(dict.fromkeys(x))
        ).reset_index()
        submission.rename(columns={'target_entity_id': 'matched_entity_ids'}, inplace=True)
    else:
        submission = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    # Crucial: Include ALL original source1 records, even those without matches (empty string)
    all_s1_ids = pd.DataFrame({'source1_entity_id': df1['entity_id']})
    submission = all_s1_ids.merge(submission, on='source1_entity_id', how='left')
    submission['matched_entity_ids'] = submission['matched_entity_ids'].fillna('')
    
    # Save submission files: matching_results.tsv (official leaderboard file), matching_result.tsv (alias requested by user) & predictions.csv
    output_tsv = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    output_tsv_alt = os.path.join(OUTPUT_DIR, 'matching_result.tsv')
    output_csv = os.path.join(OUTPUT_DIR, 'predictions.csv')
    
    submission.to_csv(output_tsv, sep='\t', index=False)
    submission.to_csv(output_tsv_alt, sep='\t', index=False)
    submission.to_csv(output_csv, index=False)
    
    logger.info("=" * 70)
    logger.info(f"  PREDICTIONS COMPLETE")
    logger.info(f"  Total Source 1 Records : {len(submission):,}")
    logger.info(f"  Records with Matches   : {len(submission[submission['matched_entity_ids'] != '']):,}")
    logger.info(f"  Records without Matches: {len(submission[submission['matched_entity_ids'] == '']):,}")
    logger.info(f"  Official Leaderboard Output : {output_tsv}")
    logger.info(f"  User Target Output          : {output_tsv_alt}")
    logger.info(f"  CSV Copy                    : {output_csv}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
