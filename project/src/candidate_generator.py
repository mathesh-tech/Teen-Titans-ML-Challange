import pandas as pd
import logging
import csv
import time
import os
from rapidfuzz import fuzz

try:
    import psutil
except ImportError:
    psutil = None

from collections import defaultdict
from typing import Set, List, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CandidateGenerator:
    def __init__(self, chunk_size: int = 100000, max_candidates_per_entity: int = 50):
        self.chunk_size = chunk_size
        self.max_candidates_per_entity = max_candidates_per_entity
        self.total_blocking_groups = 0
        self.skip_bucket_threshold = 500

    def _get_first_token(self, name: str) -> str:
        if pd.isna(name): return ""
        tokens = str(name).strip().split()
        return tokens[0] if tokens else ""

    def _get_first_3_chars(self, name: str) -> str:
        if pd.isna(name): return ""
        name = str(name).strip()
        return name[:3] if len(name) >= 3 else name

    def _get_token_count_bucket(self, name: str) -> str:
        if pd.isna(name): return "0"
        count = len(str(name).strip().split())
        return str(count)

    def _get_tokens(self, name: str) -> Set[str]:
        if pd.isna(name): return set()
        return set(str(name).split())

    def _build_source1_index(self, df: pd.DataFrame, id_col: str = 'entity_id') -> dict:
        block_index = {
            'b1': defaultdict(list),
            'b2': defaultdict(list),
            'b3': defaultdict(list)
        }
        
        logger.info(f"Building Multi-Level Blocking Index for Source 1 ({len(df):,} records)...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing S1"):
            country = str(row.get('country', '')).strip()
            name = str(row.get('business_name', '')).strip()
            
            if not country or not name:
                continue
                
            first_token = self._get_first_token(name)
            first_3_chars = self._get_first_3_chars(name)
            token_bucket = self._get_token_count_bucket(name)
            
            k1 = f"{country}_{first_token}"
            k2 = f"{country}_{first_3_chars}"
            k3 = f"{country}_{token_bucket}"
            
            record = {
                'id': row[id_col],
                'name': name,
                'tokens': self._get_tokens(name)
            }
            
            if first_token: block_index['b1'][k1].append(record)
            if first_3_chars: block_index['b2'][k2].append(record)
            block_index['b3'][k3].append(record)
            
        # Prune large buckets
        pruned_b1, pruned_b2, pruned_b3 = 0, 0, 0
        
        for k in list(block_index['b1'].keys()):
            if len(block_index['b1'][k]) > self.skip_bucket_threshold:
                del block_index['b1'][k]
                pruned_b1 += 1
                
        for k in list(block_index['b2'].keys()):
            if len(block_index['b2'][k]) > self.skip_bucket_threshold:
                del block_index['b2'][k]
                pruned_b2 += 1
                
        for k in list(block_index['b3'].keys()):
            if len(block_index['b3'][k]) > self.skip_bucket_threshold:
                del block_index['b3'][k]
                pruned_b3 += 1
                
        logger.info(f"Pruned buckets > {self.skip_bucket_threshold} records: B1({pruned_b1}), B2({pruned_b2}), B3({pruned_b3})")
            
        self.total_blocking_groups = len(block_index['b1']) + len(block_index['b2']) + len(block_index['b3'])
        logger.info(f"Total Blocking Groups across 3 levels: {self.total_blocking_groups:,}")
        return block_index

    def _probe_and_write(self, df_target: pd.DataFrame, block_index: dict, 
                         output_csv: str, target_name: str, mode: str = 'a', 
                         write_header: bool = False, target_id_col: str = 'entity_id') -> int:
        
        total_matches = 0
        pairs_since_last_flush = 0
        flush_threshold = 10000
        start_time = time.time()
        process = psutil.Process(os.getpid()) if psutil else None
        
        with open(output_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['source1_entity_id', 'target_entity_id'])
                f.flush()
            
            for start_idx in tqdm(range(0, len(df_target), self.chunk_size), desc=f"Probing {target_name}"):
                end_idx = min(start_idx + self.chunk_size, len(df_target))
                chunk = df_target.iloc[start_idx:end_idx]
                
                chunk_pairs = []
                
                for _, row in chunk.iterrows():
                    country = str(row.get('country', '')).strip()
                    name = str(row.get('business_name', '')).strip()
                    if not country or not name: continue
                    
                    target_tokens = self._get_tokens(name)
                    if not target_tokens: continue
                        
                    target_id = row[target_id_col]
                    
                    k1 = f"{country}_{self._get_first_token(name)}"
                    k2 = f"{country}_{self._get_first_3_chars(name)}"
                    k3 = f"{country}_{self._get_token_count_bucket(name)}"
                    
                    candidates_to_check = []
                    if k1 in block_index['b1']: candidates_to_check.extend(block_index['b1'][k1])
                    if k2 in block_index['b2']: candidates_to_check.extend(block_index['b2'][k2])
                    if k3 in block_index['b3']: candidates_to_check.extend(block_index['b3'][k3])
                    
                    seen_s1_ids = set()
                    scored_candidates = []
                    
                    for match in candidates_to_check:
                        s1_id = match['id']
                        if s1_id in seen_s1_ids: continue
                        seen_s1_ids.add(s1_id)
                        
                        score = fuzz.token_sort_ratio(name, match['name'])
                        scored_candidates.append((score, s1_id))
                        
                    if scored_candidates:
                        # Sort descending by score, keep top N
                        scored_candidates.sort(key=lambda x: x[0], reverse=True)
                        top_candidates = scored_candidates[:self.max_candidates_per_entity]
                        
                        for score, s1_id in top_candidates:
                            chunk_pairs.append((s1_id, target_id))
                
                if chunk_pairs:
                    writer.writerows(chunk_pairs)
                    added = len(chunk_pairs)
                    total_matches += added
                    pairs_since_last_flush += added
                    
                    if pairs_since_last_flush >= flush_threshold:
                        f.flush()
                        pairs_since_last_flush = 0
                        
                # ── Detailed Logging ──
                elapsed_mins = (time.time() - start_time) / 60.0
                if elapsed_mins > 0:
                    cps = total_matches / elapsed_mins
                    pct_done = end_idx / len(df_target)
                    est_total_mins = elapsed_mins / pct_done if pct_done > 0 else 0
                    eta_mins = est_total_mins - elapsed_mins
                    
                    mem_mb = process.memory_info().rss / (1024 ** 2) if process else 0
                    logger.info(
                        f"\n[STEP] Candidate Generation | "
                        f"Records processed: {end_idx:,} | "
                        f"Candidates generated: {total_matches:,} | "
                        f"Rate: {cps:,.0f}/min | "
                        f"ETA: {eta_mins:.1f} min\n"
                        f"[MEMORY] {mem_mb/1024:.1f} GB used"
                    )
                    
        return total_matches

    def generate(self, df_src1: pd.DataFrame, df_src2: pd.DataFrame, df_src3: pd.DataFrame, 
                 output_path: str) -> pd.DataFrame:
        logger.info("Starting Multi-Level Candidate Generation...")
        
        s1_index = self._build_source1_index(df_src1, id_col='entity_id')
        
        matches_s2 = self._probe_and_write(df_src2, s1_index, output_path, "Source 2", mode='w', write_header=True)
        matches_s3 = self._probe_and_write(df_src3, s1_index, output_path, "Source 3", mode='a', write_header=False)
        
        total = matches_s2 + matches_s3
        logger.info(f"Total candidate pairs generated: {total:,}")
        
        logger.info(f"Multi-level generation complete. Pairs streamed safely to disk.")
        return pd.DataFrame()
