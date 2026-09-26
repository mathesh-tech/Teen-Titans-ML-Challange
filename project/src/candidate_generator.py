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
    def __init__(self, chunk_size: int = 100000, max_candidates_per_entity: int = 150):
        self.chunk_size = chunk_size
        self.max_candidates_per_entity = max_candidates_per_entity
        self.skip_bucket_threshold = 20000

    def _get_street_number(self, address: str) -> str:
        if pd.isna(address): return ""
        import re
        match = re.search(r'\b\d+\b', str(address))
        return match.group(0) if match else ""

    def _build_source1_index(self, df: pd.DataFrame, id_col: str = 'entity_id') -> dict:
        block_index = {
            'tokens': defaultdict(list),
            'prefixes': defaultdict(list),
            'streets': defaultdict(list)
        }
        
        logger.info(f"Building Inverted Index for Source 1 ({len(df):,} records)...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing S1", disable=True):
            country = str(row.get('country', '')).strip()
            name = str(row.get('business_name', '')).strip()
            address = str(row.get('business_address', '')).strip()
            
            if not country or not name:
                continue
                
            record = {'id': row[id_col], 'name': name}
            
            # Block 1: All Tokens >= 3 chars
            tokens = set([t for t in name.split() if len(t) >= 3])
            for token in tokens:
                k1 = f"{country}_{token}"
                block_index['tokens'][k1].append(record)
                
            # Block 2: Prefix
            prefix = name[:5]
            if len(prefix) >= 3:
                k2 = f"{country}_{prefix}"
                block_index['prefixes'][k2].append(record)
                
            # Block 3: Street Number
            street_number = self._get_street_number(address)
            if street_number:
                k3 = f"{country}_{street_number}"
                block_index['streets'][k3].append(record)
            
        # Prune large buckets
        pruned = 0
        for block_type in block_index.keys():
            for k in list(block_index[block_type].keys()):
                if len(block_index[block_type][k]) > self.skip_bucket_threshold:
                    del block_index[block_type][k]
                    pruned += 1
                
        logger.info(f"Pruned {pruned} massive buckets > {self.skip_bucket_threshold} records.")
        logger.info(f"Total Tokens: {len(block_index['tokens']):,}")
        logger.info(f"Total Prefixes: {len(block_index['prefixes']):,}")
        logger.info(f"Total Streets: {len(block_index['streets']):,}")
        
        return block_index

    def _probe_and_write(self, df_target: pd.DataFrame, block_index: dict, 
                         output_csv: str, target_name: str, mode: str = 'a', 
                         write_header: bool = False, target_id_col: str = 'entity_id') -> int:
        
        total_matches = 0
        pairs_since_last_flush = 0
        flush_threshold = 10000
        start_time = time.time()
        process = psutil.Process(os.getpid()) if psutil else None
        
        delim = '\t' if output_csv.endswith('.tsv') else ','
        with open(output_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delim)
            if write_header:
                writer.writerow(['source1_entity_id', 'target_entity_id'])
                f.flush()
            
            for start_idx in tqdm(range(0, len(df_target), self.chunk_size), desc=f"Probing {target_name}", disable=True):
                end_idx = min(start_idx + self.chunk_size, len(df_target))
                chunk = df_target.iloc[start_idx:end_idx]
                
                chunk_pairs = []
                
                for _, row in chunk.iterrows():
                    country = str(row.get('country', '')).strip()
                    name = str(row.get('business_name', '')).strip()
                    address = str(row.get('business_address', '')).strip()
                    if not country or not name: continue
                        
                    target_id = row[target_id_col]
                    
                    candidates_to_check = []
                    
                    # Probe Tokens
                    tokens = set([t for t in name.split() if len(t) >= 3])
                    for token in tokens:
                        k1 = f"{country}_{token}"
                        if k1 in block_index['tokens']:
                            candidates_to_check.extend(block_index['tokens'][k1])
                            
                    # Probe Prefix
                    prefix = name[:5]
                    if len(prefix) >= 3:
                        k2 = f"{country}_{prefix}"
                        if k2 in block_index['prefixes']:
                            candidates_to_check.extend(block_index['prefixes'][k2])
                            
                    # Probe Street Number
                    street_number = self._get_street_number(address)
                    if street_number:
                        k3 = f"{country}_{street_number}"
                        if k3 in block_index['streets']:
                            candidates_to_check.extend(block_index['streets'][k3])
                    
                    seen_s1_ids = set()
                    scored_candidates = []
                    
                    for match in candidates_to_check:
                        s1_id = match['id']
                        if s1_id in seen_s1_ids: continue
                        seen_s1_ids.add(s1_id)
                        
                        # Only calculate fuzz if necessary, or just calculate it
                        score = fuzz.token_sort_ratio(name, match['name'])
                        
                        # Apply a loose pre-filter to drop absolute garbage (score < 30)
                        if score >= 30:
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
                        f"[MEMORY] {mem_mb/1024:.1f} MB used"
                    )
                    
        return total_matches

    def generate(self, df_src1: pd.DataFrame, df_src2: pd.DataFrame, df_src3: pd.DataFrame, 
                 output_path: str) -> pd.DataFrame:
        logger.info("Starting Inverted-Index Candidate Generation...")
        
        s1_index = self._build_source1_index(df_src1, id_col='entity_id')
        
        matches_s2 = self._probe_and_write(df_src2, s1_index, output_path, "Source 2", mode='w', write_header=True)
        matches_s3 = self._probe_and_write(df_src3, s1_index, output_path, "Source 3", mode='a', write_header=False)
        
        total = matches_s2 + matches_s3
        logger.info(f"Total candidate pairs generated: {total:,}")
        
        logger.info(f"Inverted-Index generation complete. Pairs streamed safely to disk.")
        return pd.DataFrame()
