"""
pipeline.py
-----------
Master end-to-end pipeline for the Amazon ML Challenge 2026.
Business Entity Resolution: Source1 <-> Source2 + Source3

Flow:
  1. Load Data        (DataLoader)
  2. Clean Data       (DataCleaner)
  3. Generate Candidates (CandidateGenerator)
  4. Generate Features   (FeatureEngineer)
  5. Build Training Dataset (TrainingPairGenerator)
  6. Train Model      (ModelTrainer)
  7. Optimize Threshold (ThresholdOptimizer)
"""

import os
import sys
import logging
import pandas as pd

# Ensure src/ is on path when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader           import DataLoader
from data_cleaner          import DataCleaner
from candidate_generator   import CandidateGenerator
from feature_engineer      import FeatureEngineer
from training_pair_generator import TrainingPairGenerator
from model_trainer         import ModelTrainer
from threshold_optimizer   import ThresholdOptimizer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('pipeline')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


def run_pipeline():
    logger.info("=" * 70)
    logger.info("  Amazon ML Challenge 2026 — Business Entity Resolution Pipeline")
    logger.info("=" * 70)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    logger.info("STEP 1: Loading datasets...")
    loader = DataLoader(data_dir=os.path.join(BASE_DIR, 'dataset'))
    df1 = loader.load_source1()
    df2 = loader.load_source2()
    df3 = loader.load_source3()
    gt  = loader.load_ground_truth()
    logger.info(f"  Source1: {df1.shape}  Source2: {df2.shape}  Source3: {df3.shape}  GT: {gt.shape}")

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    logger.info("STEP 2: Cleaning data...")
    cleaner = DataCleaner()
    df1_clean = cleaner.process(df1)
    df2_clean = cleaner.process(df2)
    df3_clean = cleaner.process(df3)

    # Combine Source2 + Source3 into a single target pool
    df_target = pd.concat([df2_clean, df3_clean], ignore_index=True)
    logger.info(f"  Combined target pool: {df_target.shape}")

    # ── Step 3: Generate Candidates ──────────────────────────────────────────
    logger.info("STEP 3: Generating candidate pairs...")
    candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.csv')
    gen = CandidateGenerator()
    candidates = gen.generate(
        df_src1=df1_clean,
        df_src2=df2_clean,
        df_src3=df3_clean,
        output_path=candidate_path
    )
    logger.info(f"  Candidate pairs: {len(candidates)}")

    if candidates.empty:
        logger.error("No candidates generated. Exiting pipeline.")
        sys.exit(1)

    # ── Step 4 + 5: Features + Training Labels ───────────────────────────────
    logger.info("STEP 4 & 5: Building training dataset with features + labels...")
    engineer = FeatureEngineer()
    tpg = TrainingPairGenerator()

    training_dataset = tpg.generate_training_data(
        candidates_df=candidates,
        ground_truth_df=gt,
        df_source=df1_clean,
        df_target=df_target,
        feature_engineer=engineer,
        output_path=os.path.join(OUTPUT_DIR, 'training_dataset.csv')
    )
    logger.info(f"  Training dataset shape: {training_dataset.shape}")
    logger.info(f"  Label distribution:\n{training_dataset['label'].value_counts().to_string()}")

    # ── Step 6: Train Model ──────────────────────────────────────────────────
    logger.info("STEP 6: Training LightGBM model...")
    trainer = ModelTrainer()
    metrics = trainer.train_and_evaluate(
        training_data_path=os.path.join(OUTPUT_DIR, 'training_dataset.csv'),
        model_save_path=os.path.join(MODEL_DIR, 'lightgbm_model.pkl'),
        metrics_save_path=os.path.join(OUTPUT_DIR, 'training_metrics.json')
    )
    logger.info(f"  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F0.5: {metrics['f0.5_score']:.4f}")

    # ── Step 7: Optimize Threshold ───────────────────────────────────────────
    logger.info("STEP 7: Optimizing probability threshold...")
    # Get validation probabilities from the trainer for threshold optimization
    import joblib
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'training_dataset.csv'))
    exclude = ['source1_entity_id', 'target_entity_id', 'label']
    feature_cols = [c for c in df_train.columns if c not in exclude]
    X = df_train[feature_cols]
    y = df_train['label']

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(os.path.join(MODEL_DIR, 'lightgbm_model.pkl'))
    y_prob = model.predict_proba(X_val)[:, 1]

    optimizer = ThresholdOptimizer()
    best = optimizer.optimize(
        y_true=y_val.values,
        y_prob=y_prob,
        save_path=os.path.join(OUTPUT_DIR, 'best_threshold.json')
    )

    logger.info("=" * 70)
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"  Best Threshold : {best['best_threshold']}")
    logger.info(f"  Best F0.5 Score: {best['best_metrics']['f0.5_score']:.4f}")
    logger.info(f"  Best Precision : {best['best_metrics']['precision']:.4f}")
    logger.info(f"  Best Recall    : {best['best_metrics']['recall']:.4f}")
    logger.info("=" * 70)
    logger.info(f"  Artifacts saved to:")
    logger.info(f"    output/candidate_pairs.csv")
    logger.info(f"    output/training_dataset.csv")
    logger.info(f"    output/training_metrics.json")
    logger.info(f"    output/best_threshold.json")
    logger.info(f"    models/lightgbm_model.pkl")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
