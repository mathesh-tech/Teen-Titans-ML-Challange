"""
train.py
-----------
Master end-to-end pipeline for the Amazon ML Challenge 2026.
Business Entity Resolution: Source1 <-> Source2 + Source3

Features:
  - Auto-Resume Checkpoints
  - Automatic Status Detection
  - Automatic Error Recovery
"""

import os
import sys
import logging
import pandas as pd
import time
import traceback

try:
    import psutil
except ImportError:
    psutil = None

# Ensure src/ is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from data_loader           import DataLoader
from data_cleaner          import DataCleaner
from candidate_generator   import CandidateGenerator
from feature_engineer      import FeatureEngineer
from training_pair_generator import TrainingPairGenerator
from model_trainer         import ModelTrainer
from threshold_optimizer   import ThresholdOptimizer

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, 'output'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, 'output', 'pipeline.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger('pipeline')

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

start_time = time.time()
peak_memory_mb = 0.0

def log_status(stage_name, progress_pct):
    global peak_memory_mb
    elapsed = time.time() - start_time
    mem_mb = 0.0
    cpu_usage = 0.0
    if psutil:
        p = psutil.Process(os.getpid())
        mem_mb = p.memory_info().rss / (1024 ** 2)
        cpu_usage = psutil.cpu_percent(interval=None)
        if mem_mb > peak_memory_mb:
            peak_memory_mb = mem_mb
            
    logger.info(f"--- STATUS ---")
    logger.info(f"  Current Stage: {stage_name}")
    logger.info(f"  Progress:      {progress_pct}%")
    logger.info(f"  Elapsed Time:  {elapsed:.1f} sec")
    logger.info(f"  Memory Usage:  {mem_mb:.2f} MB")
    logger.info(f"  CPU Usage:     {cpu_usage}%")
    logger.info(f"--------------")


def run_pipeline():
    logger.info("=" * 70)
    logger.info("  Amazon ML Challenge 2026 — Fault-Tolerant Pipeline")
    logger.info("=" * 70)

    try:
        # ── Step 1: Load ─────────────────────────────────────────────────────────
        log_status("Data Loading", 5)
        loader = DataLoader(data_dir=os.path.join(BASE_DIR, 'dataset'))
        df1 = loader.load_source1()
        df2 = loader.load_source2()
        df3 = loader.load_source3()
        gt  = loader.load_ground_truth()
        total_records = len(df1) + len(df2) + len(df3)
        
        # ── Step 2: Clean ────────────────────────────────────────────────────────
        log_status("Data Cleaning", 10)
        cleaner = DataCleaner()
        df1_clean = cleaner.process(df1)
        df2_clean = cleaner.process(df2)
        df3_clean = cleaner.process(df3)
        df_target = pd.concat([df2_clean, df3_clean], ignore_index=True)
        
        candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.csv')
        features_path = os.path.join(OUTPUT_DIR, 'features.csv')
        training_data_path = os.path.join(OUTPUT_DIR, 'training_dataset.csv')
        model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
        metrics_path = os.path.join(OUTPUT_DIR, 'training_metrics.json')

        # ── Step 3: Generate Candidates ──────────────────────────────────────────
        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 100:
            log_status("Candidate Generation [AUTO-RESUME SKIP]", 30)
            logger.info("Auto-Resume: candidate_pairs.csv exists and is not empty. Loading from disk...")
        else:
            log_status("Candidate Generation", 15)
            gen = CandidateGenerator(chunk_size=100000, max_candidates_per_entity=20)
            gen.generate(
                df_src1=df1_clean,
                df_src2=df2_clean,
                df_src3=df3_clean,
                output_path=candidate_path
            )

        logger.info(f"Candidate file exists: {os.path.exists(candidate_path)}")
        logger.info(f"Candidate file size: {os.path.getsize(candidate_path)} bytes")
        
        # ── Phase 2: Candidate Validation ────────────────────────────────────────
        unique_s1 = set()
        unique_target = set()
        candidate_count = 0
        for chunk in pd.read_csv(candidate_path, chunksize=100000, dtype=str):
            candidate_count += len(chunk)
            unique_s1.update(chunk['source1_entity_id'].dropna())
            unique_target.update(chunk['target_entity_id'].dropna())
            
        logger.info(f"Candidate count: {candidate_count}")
            
        if candidate_count == 0:
            logger.error("No candidates generated. Exiting pipeline.")
            sys.exit(1)
            
        avg_cands = candidate_count / len(unique_s1) if len(unique_s1) > 0 else 0
        
        cand_stats = (
            f"Total candidate pairs: {candidate_count:,}\n"
            f"Unique source1 entities: {len(unique_s1):,}\n"
            f"Unique target entities: {len(unique_target):,}\n"
            f"Average candidates per source1 entity: {avg_cands:.2f}\n"
        )
        logger.info(f"\n--- CANDIDATE STATS ---\n{cand_stats}-----------------------")
        
        with open(os.path.join(OUTPUT_DIR, 'candidate_stats.txt'), 'w') as f:
            f.write(cand_stats)
            
        if candidate_count > 5000000:
            logger.error("CANDIDATE EXPLOSION DETECTED")
            logger.error("Candidate generation produced over 5,000,000 pairs. Training aborted to prevent OOM.")
            logger.error("RECOMMENDED FIX: Use stronger blocking (e.g. multi-level blocking with country+token)")
            sys.exit(1)

        # ── Step 4 & 5: Feature Engineering ──────────────────────────────────────
        feature_cols = []
        if os.path.exists(training_data_path) and os.path.getsize(training_data_path) > 100:
            log_status("Feature Engineering [AUTO-RESUME SKIP]", 60)
            logger.info("Auto-Resume: training_dataset.csv exists. Skipping FE...")
            df_train_cols = pd.read_csv(training_data_path, nrows=1)
            exclude = ['source1_entity_id', 'target_entity_id', 'label']
            feature_cols = [c for c in df_train_cols.columns if c not in exclude]
        else:
            if not (os.path.exists(features_path) and os.path.getsize(features_path) > 100):
                log_status("Feature Engineering", 40)
                engineer = FeatureEngineer(chunk_size=100000)
                engineer.generate_features(candidate_path, df1_clean, df_target, features_path)
                
            log_status("Training Dataset Generation", 50)
            tpg = TrainingPairGenerator(chunk_size=500000)
            tpg.generate_training_data(features_path, gt, training_data_path)
            
            df_train_cols = pd.read_csv(training_data_path, nrows=1)
            exclude = ['source1_entity_id', 'target_entity_id', 'label']
            feature_cols = [c for c in df_train_cols.columns if c not in exclude]
            
        # ── Phase 3: Feature Validation ────────────────────────────────────────
        if not os.path.exists(features_path):
            logger.error("features.csv does not exist!")
            sys.exit(1)
            
        feature_count = sum(1 for _ in open(features_path)) - 1
        if feature_count <= 0:
            logger.error("features.csv is empty! Aborting.")
            sys.exit(1)
            
        df_feat_cols = pd.read_csv(features_path, nrows=1)
        logger.info(f"Feature file exists: True")
        logger.info(f"Feature row count: {feature_count:,}")
        logger.info(f"Feature column count: {len(df_feat_cols.columns)}")
        logger.info(f"Feature names: {list(df_feat_cols.columns)}")
        logger.info(f"Training dataset features: {len(feature_cols)}")

        # ── Step 6: Train Model ──────────────────────────────────────────────────
        metrics = {}
        training_time = 0
        if os.path.exists(model_path) and os.path.exists(metrics_path):
            log_status("Model Training [AUTO-RESUME SKIP]", 85)
            logger.info("Auto-Resume: lightgbm_model.pkl exists. Skipping training...")
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            log_status("Model Training", 70)
            train_start = time.time()
            trainer = ModelTrainer()
            metrics = trainer.train_and_evaluate(
                training_data_path=training_data_path,
                model_save_path=model_path,
                metrics_save_path=metrics_path
            )
            training_time = time.time() - train_start

        # ── Step 7: Optimize Threshold ───────────────────────────────────────────
        log_status("Threshold Optimization", 90)
        import joblib
        from sklearn.model_selection import train_test_split
        
        df_train = pd.read_csv(training_data_path)
        exclude = ['source1_entity_id', 'target_entity_id', 'label']
        X = df_train[[c for c in df_train.columns if c not in exclude]]
        y = df_train['label']
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = joblib.load(model_path)
        y_prob = model.predict(X_val) # lightgbm native API returns probabilities
        
        optimizer = ThresholdOptimizer()
        best = optimizer.optimize(
            y_true=y_val.values,
            y_prob=y_prob,
            save_path=os.path.join(OUTPUT_DIR, 'best_threshold.json')
        )
        
        log_status("Completed", 100)

        # ── Step 8: Final Report & Verification ──────────────────────────────────
        run_summary_path = os.path.join(OUTPUT_DIR, 'run_summary.txt')
        
        peak_mb = peak_memory_mb
        if psutil:
            p_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            if p_mem > peak_mb: peak_mb = p_mem
            
        summary_text = (
            f"PIPELINE COMPLETED SUCCESSFULLY\n"
            f"===============================\n"
            f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes\n"
            f"Peak memory usage: {peak_mb:.1f} MB\n"
            f"Candidate count: {candidate_count:,}\n"
            f"Feature count: {len(feature_cols)}\n"
            f"Final model metrics:\n"
            f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
            f"  Precision: {metrics.get('precision', 0):.4f}\n"
            f"  Recall:    {metrics.get('recall', 0):.4f}\n"
            f"  F1 Score:  {metrics.get('f1_score', 0):.4f}\n"
            f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}\n"
        )
        
        with open(run_summary_path, 'w') as f:
            f.write(summary_text)
            
        logger.info(f"\n{summary_text}")
        
        # Verify all output files
        expected_files = [candidate_path, features_path, training_data_path, metrics_path, model_path]
        missing = [f for f in expected_files if not os.path.exists(f)]
        
        if not missing:
            final_msg = (
                "================================================\n"
                "PIPELINE COMPLETED SUCCESSFULLY\n"
                "================================================\n\n"
                "Artifacts Generated:\n\n"
                "✓ candidate_pairs.csv\n"
                "✓ features.csv\n"
                "✓ training_dataset.csv\n"
                "✓ training_metrics.json\n"
                "✓ run_summary.txt\n"
                "✓ lightgbm_model.pkl\n\n"
                "================================================"
            )
            logger.info(f"\n{final_msg}")
        else:
            logger.error("PIPELINE FINISHED BUT FILES ARE MISSING:")
            for m in missing:
                logger.error(f"  Missing: {m}")

    except Exception as e:
        error_log = os.path.join(OUTPUT_DIR, 'error.log')
        with open(error_log, 'w') as f:
            f.write(traceback.format_exc())
        logger.error(f"PIPELINE FAILED! Exception caught. Stack trace saved to {error_log}")
        logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
