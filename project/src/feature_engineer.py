import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import time

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, chunk_size=100000):
        self.chunk_size = chunk_size
        self.tfidf_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.tfidf_address = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.is_fitted = False
        
    def _fit_tfidf(self, df_source: pd.DataFrame, df_target: pd.DataFrame):
        logger.info("Fitting TF-IDF on a representative sample to save time/RAM...")
        # Sampling 500k records is highly sufficient to learn character n-gram frequencies
        sample_src = df_source.sample(min(250000, len(df_source)), random_state=42)
        sample_tgt = df_target.sample(min(250000, len(df_target)), random_state=42)
        
        all_names = pd.concat([sample_src['business_name'], sample_tgt['business_name']]).fillna('')
        all_addrs = pd.concat([sample_src['business_address'], sample_tgt['business_address']]).fillna('')
        
        self.tfidf_name.fit(all_names)
        self.tfidf_address.fit(all_addrs)
        self.is_fitted = True
        logger.info("TF-IDF fitting complete.")

    def _extract_street_number(self, address: str) -> str:
        import re
        match = re.search(r'\b\d+\b', address)
        return match.group(0) if match else ""

    def _compute_features_chunk(self, chunk: pd.DataFrame, df_source: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        features = []
        
        # 1. High-Performance Pandas Merge (Replaces memory-crashing Python dictionaries)
        merged = chunk.merge(df_source, left_on='source1_entity_id', right_on='entity_id', how='left')
        merged = merged.merge(df_target, left_on='target_entity_id', right_on='entity_id', suffixes=('_s1', '_tgt'), how='left')
        
        s1_ids = merged['source1_entity_id'].tolist()
        tgt_ids = merged['target_entity_id'].tolist()
        
        s_names = merged['business_name_s1'].fillna('').astype(str).str.lower().str.strip().tolist()
        t_names = merged['business_name_tgt'].fillna('').astype(str).str.lower().str.strip().tolist()
        
        s_addrs = merged['business_address_s1'].fillna('').astype(str).str.lower().str.strip().tolist()
        t_addrs = merged['business_address_tgt'].fillna('').astype(str).str.lower().str.strip().tolist()
        
        s_countries = merged['country_s1'].fillna('').astype(str).str.lower().str.strip().tolist()
        t_countries = merged['country_tgt'].fillna('').astype(str).str.lower().str.strip().tolist()
        
        # 2. Tokenization
        n1_tokens = [set(n.split()) for n in s_names]
        n2_tokens = [set(n.split()) for n in t_names]
        a1_tokens = [set(a.split()) for a in s_addrs]
        a2_tokens = [set(a.split()) for a in t_addrs]
        
        # 3. Vectorized feature generation via list comprehensions
        for i in range(len(s1_ids)):
            n1, n2 = s_names[i], t_names[i]
            a1, a2 = s_addrs[i], t_addrs[i]
            c1, c2 = s_countries[i], t_countries[i]
            
            n1_tok, n2_tok = n1_tokens[i], n2_tokens[i]
            a1_tok, a2_tok = a1_tokens[i], a2_tokens[i]
            
            # Name Features
            exact_match = 1 if n1 == n2 and n1 != "" else 0
            token_overlap = len(n1_tok.intersection(n2_tok))
            jaccard = token_overlap / len(n1_tok.union(n2_tok)) if n1_tok.union(n2_tok) else 0.0
            fuzz_ratio = fuzz.ratio(n1, n2) / 100.0
            token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
            partial = fuzz.partial_ratio(n1, n2) / 100.0
            length_diff = abs(len(n1) - len(n2))
            name_contains = 1 if (n1 in n2 or n2 in n1) and n1 != "" and n2 != "" else 0
            
            # Address Features
            addr_overlap = len(a1_tok.intersection(a2_tok))
            addr_jaccard = addr_overlap / len(a1_tok.union(a2_tok)) if a1_tok.union(a2_tok) else 0.0
            addr_len_diff = abs(len(a1) - len(a2))
            
            # New Address features
            num1 = self._extract_street_number(a1)
            num2 = self._extract_street_number(a2)
            street_num_match = 1 if num1 == num2 and num1 != "" else (0 if num1 and num2 else -1)
            
            # Country
            same_country = 1 if c1 == c2 and c1 != "" else 0
            
            features.append([
                s1_ids[i], tgt_ids[i],
                exact_match, token_overlap, jaccard, fuzz_ratio, token_sort, partial, length_diff, name_contains,
                addr_overlap, addr_jaccard, addr_len_diff, street_num_match,
                same_country
            ])
            
        df_feats = pd.DataFrame(features, columns=[
            'source1_entity_id', 'target_entity_id',
            'exact_match', 'token_overlap', 'jaccard_similarity', 'fuzz_ratio',
            'token_sort_ratio', 'partial_ratio', 'length_difference', 'name_contains_match',
            'address_overlap', 'address_jaccard', 'address_length_difference', 'street_number_match',
            'same_country'
        ])
        
        # Compute TF-IDF similarities
        if self.is_fitted and len(chunk) > 0:
            vec_n1 = self.tfidf_name.transform(s_names)
            vec_n2 = self.tfidf_name.transform(t_names)
            vec_a1 = self.tfidf_address.transform(s_addrs)
            vec_a2 = self.tfidf_address.transform(t_addrs)
            
            # Fast sparse dot product
            cos_n = vec_n1.multiply(vec_n2).sum(axis=1)
            cos_a = vec_a1.multiply(vec_a2).sum(axis=1)
            
            df_feats['name_tfidf_similarity'] = np.asarray(cos_n).flatten()
            df_feats['address_tfidf_similarity'] = np.asarray(cos_a).flatten()
        else:
            df_feats['name_tfidf_similarity'] = 0.0
            df_feats['address_tfidf_similarity'] = 0.0
            
        return df_feats

    def generate_features(self, candidates_path: str, df_source: pd.DataFrame, df_target: pd.DataFrame, 
                          output_path: str) -> pd.DataFrame:
        
        required_columns = ["entity_id", "business_name", "business_address", "country"]
        
        for name, df in [("Source", df_source), ("Target", df_target)]:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                logger.error(f"Missing columns in {name} dataset: {missing}")
                raise ValueError(f"Missing required columns: {missing}")
        
        if not self.is_fitted:
            self._fit_tfidf(df_source, df_target)
            
        logger.info(f"Generating features in chunks of {self.chunk_size} to save RAM...")
        first_chunk = True
        total_chunks = 0
        total_rows = 0
        start_time = time.time()
        
        total_candidate_rows = sum(1 for _ in open(candidates_path)) - 1 if os.path.exists(candidates_path) else 0
            
        for chunk in pd.read_csv(candidates_path, chunksize=self.chunk_size, dtype=str):
            feat_chunk = self._compute_features_chunk(chunk, df_source, df_target)
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            feat_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False
            total_chunks += 1
            total_rows += len(feat_chunk)
            
            elapsed = time.time() - start_time
            rate = total_rows / elapsed * 60 if elapsed > 0 else 0
            eta = (total_candidate_rows - total_rows) / rate if rate > 0 else 0
                
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) if psutil else 0
            logger.info(
                f"[FE STEP] Chunk {total_chunks} | "
                f"Processed: {total_rows}/{total_candidate_rows} | "
                f"Rate: {rate:,.0f} rows/min | "
                f"ETA: {eta:.1f} min | "
                f"RAM: {mem_mb/1024:.1f} GB"
            )
            
        logger.info(f"Feature engineering complete. Total rows processed: {total_rows}")
        return pd.DataFrame()
