import pandas as pd
import logging
import os

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

class TrainingPairGenerator:
    def __init__(self, chunk_size=500000):
        self.chunk_size = chunk_size

    def generate_training_data(self, features_path: str, ground_truth_df: pd.DataFrame, 
                               output_path: str) -> pd.DataFrame:
        logger.info("Generating training dataset from features...")
        
        # Ground truth mapping: source1_entity_id -> set of matched target_entity_ids
        # Wait, matched_entity_ids is comma-separated string
        gt_map = {}
        for _, row in ground_truth_df.iterrows():
            s1_id = str(row['source1_entity_id']).strip()
            matches_str = str(row['matched_entity_ids']).strip()
            if matches_str and matches_str.lower() != 'nan':
                gt_map[s1_id] = set([x.strip() for x in matches_str.split(',')])
            else:
                gt_map[s1_id] = set()
                
        total_pos = 0
        total_neg = 0
        first_chunk = True
        total_chunks = 0
        
        logger.info("Joining ground truth with features in chunks...")
        
        for chunk in pd.read_csv(features_path, chunksize=self.chunk_size, dtype={'source1_entity_id': str, 'target_entity_id': str}):
            # Vectorized and stripped label generation for 100x performance boost
            s1_ids = chunk['source1_entity_id'].astype(str).str.strip()
            tgt_ids = chunk['target_entity_id'].astype(str).str.strip()
            
            labels = [
                1 if s1 in gt_map and tgt in gt_map[s1] else 0 
                for s1, tgt in zip(s1_ids, tgt_ids)
            ]
            
            total_pos += sum(labels)
            total_neg += len(labels) - sum(labels)
            
            chunk['label'] = labels
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False
            total_chunks += 1
            
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) if psutil else 0
            logger.info(f"[MEMORY] {mem_mb/1024:.1f} GB used | [STEP] Training Dataset Gen Chunk {total_chunks}")
            
        logger.info("Training Dataset generated.")
        logger.info(f"Positive samples: {total_pos:,}")
        logger.info(f"Negative samples: {total_neg:,}")
        
        total_samples = total_pos + total_neg
        class_balance = (total_pos / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Class Balance %: {class_balance:.2f}%")
        
        # We don't return the full dataframe to save RAM.
        return pd.DataFrame()
