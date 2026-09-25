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
import pandas as pd
from pathlib import Path
from typing import Tuple, Any

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

    # 4. Generate Candidates
    logger.info("STEP 2: Generating candidate pairs for test data...")
    gen = CandidateGenerator()
    candidates = gen.generate(
        df_src1=df1_clean,
        df_src2=df2_clean,
        df_src3=df3_clean,
        output_path=os.path.join(OUTPUT_DIR, 'test_candidate_pairs.csv')
    )
    logger.info(f"  Generated {len(candidates)} candidate pairs.")

    if candidates.empty:
        logger.error("No candidates generated for test data. Cannot proceed with predictions.")
        sys.exit(1)

    # 5. Generate Features
    logger.info("STEP 3: Generating features for test candidates...")
    engineer = FeatureEngineer()
    
    # Note: In a strict pipeline, TF-IDF vectorizers should be loaded from training.
    # Here, fitting on the test set acts as transductive learning, which adapts to test distribution.
    features_df = engineer.generate_features(
        df_pairs=candidates,
        df_source=df1_clean,
        df_target=df_target,
        src_id_col='source1_entity_id',
        tgt_id_col='target_entity_id'
    )
    
    # 6. Predict Probabilities
    logger.info("STEP 4: Predicting match probabilities...")
    # Select only feature columns, excluding IDs
    exclude_cols = ['source1_entity_id', 'target_entity_id', 'label']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    X_test = features_df[feature_cols].astype(float)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 7. Apply Threshold & Generate Output
    logger.info(f"STEP 5: Applying optimal threshold ({threshold})...")
    predictions = (y_prob >= threshold).astype(int)
    
    # Add predictions to candidate pairs
    results_df = candidates.copy()
    results_df['match_probability'] = y_prob
    results_df['prediction'] = predictions
    
    # Filter for positive predictions
    positive_matches = results_df[results_df['prediction'] == 1].copy()
    
    logger.info("STEP 6: Formatting predictions for submission...")
    # Group by source1_entity_id and join target_entity_id with commas (one-to-many format)
    submission = positive_matches.groupby('source1_entity_id')['target_entity_id'].apply(
        lambda x: ','.join(x)
    ).reset_index()
    submission.rename(columns={'target_entity_id': 'matched_entity_ids'}, inplace=True)
    
    # Crucial: Include all original source1 records, even those without matches (empty string)
    all_s1_ids = pd.DataFrame({'source1_entity_id': df1['entity_id']})
    submission = all_s1_ids.merge(submission, on='source1_entity_id', how='left')
    submission['matched_entity_ids'] = submission['matched_entity_ids'].fillna('')
    
    # Save predictions
    output_file = os.path.join(OUTPUT_DIR, 'predictions.csv')
    submission.to_csv(output_file, index=False)
    
    logger.info("=" * 70)
    logger.info(f"  PREDICTIONS COMPLETE")
    logger.info(f"  Total Source 1 Records : {len(submission)}")
    logger.info(f"  Records with Matches   : {len(submission[submission['matched_entity_ids'] != ''])}")
    logger.info(f"  Records without Matches: {len(submission[submission['matched_entity_ids'] == ''])}")
    logger.info(f"  Output saved to        : {output_file}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
