import pandas as pd
import logging
import csv
import time
import os
import re
from rapidfuzz import fuzz

try:
    import psutil
except ImportError:
    psutil = None

from collections import defaultdict
from typing import Set, List, Dict

logger = logging.getLogger(__name__)

STOP_WORDS = {
    # English corporate & generic words
    'private', 'limited', 'corporation', 'company', 'corp', 'inc', 'llc',
    'pvt', 'ltd', 'the', 'and', 'for', 'group', 'services', 'solutions',
    'enterprises', 'industries', 'international', 'technologies', 'india',
    'associates', 'partners', 'ventures', 'consultants', 'holdings', 'trading',
    'brothers', 'centre', 'center', 'foundation', 'management', 'consulting',
    'system', 'systems', 'agency', 'logistics', 'global',
    # French corporate & generic words
    'sarl', 'sas', 'eurl', 'sci', 'snc', 'france', 'ste', 'societe', 'ets',
    'etablissement', 'groupe', 'association', 'syndicat', 'cooperative', 'mutuelle'
}

PINCODE_RE = re.compile(r'\b\d{5,6}\b')
TOKEN_RE = re.compile(r'[a-z0-9]+')

class CandidateGenerator:
    def __init__(self, chunk_size: int = 100000, max_candidates_per_entity: int = 6):
        self.chunk_size = chunk_size
        self.max_candidates_per_entity = max_candidates_per_entity
        self.bucket_cap = 1500

    def _build_source1_index(self, df: pd.DataFrame, id_col: str = 'entity_id') -> dict:
        block_index = defaultdict(list)
        
        logger.info(f"Building Enhanced Inverted Index for Source 1 ({len(df):,} records)...")
        
        for row in df.itertuples(index=False):
            country = str(getattr(row, 'country', '')).strip()
            name = str(getattr(row, 'business_name', '')).strip().lower()
            addr = str(getattr(row, 'business_address', '')).strip().lower()
            
            if not country or not name:
                continue
                
            s1_id = getattr(row, id_col)
            record = {'id': s1_id, 'name': name}
            tokens = [t for t in TOKEN_RE.findall(name) if len(t) >= 3 and t not in STOP_WORDS]
            
            # Key 1: Bigram token key (Extremely high precision & distinctiveness)
            if len(tokens) >= 2:
                block_index[f"{country}_{tokens[0]}_{tokens[1]}"].append(record)
                
            # Key 2: Distinct Unigram tokens (up to 2 tokens)
            for t in tokens[:2]:
                block_index[f"{country}_{t}"].append(record)
                
            # Key 3: Pincode / Zipcode + primary token
            pincode_match = PINCODE_RE.search(addr)
            if pincode_match and tokens:
                block_index[f"{country}_{pincode_match.group(0)}_{tokens[0]}"].append(record)
                
            # Key 4: 4-character prefix
            if len(name) >= 4 and not name.startswith(tuple(STOP_WORDS)):
                block_index[f"{country}_{name[:4]}"].append(record)
            
        # Cap oversized buckets to prevent combinatorial explosion without dropping entities
        capped = 0
        for k in list(block_index.keys()):
            if len(block_index[k]) > self.bucket_cap:
                block_index[k] = block_index[k][:self.bucket_cap]
                capped += 1
                
        logger.info(f"Capped {capped} large buckets to max {self.bucket_cap} records.")
        logger.info(f"Total Blocking Keys in Index: {len(block_index):,}")
        
        return block_index

    def _probe_and_write(self, df_target: pd.DataFrame, block_index: dict, 
                         output_csv: str, target_name: str, mode: str = 'a', 
                         write_header: bool = False, target_id_col: str = 'entity_id') -> int:
        
        total_matches = 0
        pairs_since_last_flush = 0
        flush_threshold = 20000
        start_time = time.time()
        process = psutil.Process(os.getpid()) if psutil else None
        
        delim = '\t' if output_csv.endswith('.tsv') else ','
        with open(output_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delim)
            if write_header:
                writer.writerow(['source1_entity_id', 'target_entity_id'])
                f.flush()
            
            for start_idx in range(0, len(df_target), self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, len(df_target))
                chunk = df_target.iloc[start_idx:end_idx]
                
                chunk_pairs = []
                
                for row in chunk.itertuples(index=False):
                    country = str(getattr(row, 'country', '')).strip()
                    name = str(getattr(row, 'business_name', '')).strip().lower()
                    addr = str(getattr(row, 'business_address', '')).strip().lower()
                    if not country or not name:
                        continue
                        
                    target_id = getattr(row, target_id_col)
                    tokens = [t for t in TOKEN_RE.findall(name) if len(t) >= 3 and t not in STOP_WORDS]
                    
                    pool = []
                    # 1. Probe Bigram
                    if len(tokens) >= 2:
                        k_bi = f"{country}_{tokens[0]}_{tokens[1]}"
                        if k_bi in block_index:
                            pool.extend(block_index[k_bi])
                            
                    # 2. Probe Pincode
                    pincode_match = PINCODE_RE.search(addr)
                    if pincode_match and tokens:
                        k_pin = f"{country}_{pincode_match.group(0)}_{tokens[0]}"
                        if k_pin in block_index:
                            pool.extend(block_index[k_pin])
                            
                    # 3. Probe Unigram tokens
                    for t in tokens[:2]:
                        if len(pool) >= 80:
                            break
                        k_u = f"{country}_{t}"
                        if k_u in block_index:
                            pool.extend(block_index[k_u])
                            
                    # 4. Probe Prefix
                    if len(pool) < 20 and len(name) >= 4:
                        k_pre = f"{country}_{name[:4]}"
                        if k_pre in block_index:
                            pool.extend(block_index[k_pre])
                    
                    if pool:
                        seen_s1_ids = set()
                        scored_candidates = []
                        for match in pool:
                            s1_id = match['id']
                            if s1_id in seen_s1_ids:
                                continue
                            seen_s1_ids.add(s1_id)
                            
                            m_name = match['name']
                            if name == m_name:
                                score = 100.0
                            else:
                                score = fuzz.token_sort_ratio(name, m_name, score_cutoff=45)
                                
                            if score >= 45:
                                scored_candidates.append((score, s1_id))
                        
                        if scored_candidates:
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
        logger.info("Starting High-Recall Inverted-Index Candidate Generation...")
        
        s1_index = self._build_source1_index(df_src1, id_col='entity_id')
        
        matches_s2 = self._probe_and_write(df_src2, s1_index, output_path, "Source 2", mode='w', write_header=True)
        matches_s3 = self._probe_and_write(df_src3, s1_index, output_path, "Source 3", mode='a', write_header=False)
        
        total = matches_s2 + matches_s3
        logger.info(f"Total candidate pairs generated: {total:,}")
        
        logger.info(f"High-Recall Inverted-Index generation complete. Pairs streamed safely to disk.")
        return pd.DataFrame()
