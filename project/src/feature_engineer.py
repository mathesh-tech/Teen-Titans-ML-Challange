import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
import os

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
        
    def _fit_tfidf(self, df_source, df_target):
        logger.info("Fitting TF-IDF on combined Source and Target...")
        all_names = pd.concat([df_source['business_name'], df_target['business_name']]).fillna('')
        all_addrs = pd.concat([df_source['business_address'], df_target['business_address']]).fillna('')
        
        self.tfidf_name.fit(all_names)
        self.tfidf_address.fit(all_addrs)
        self.is_fitted = True
        logger.info("TF-IDF fitting complete.")

    def _compute_features_chunk(self, chunk, source_map, target_map):
        features = []
        
        # 1. Vectorized Lookup
        s1_ids = chunk['source1_entity_id'].values
        tgt_ids = chunk['target_entity_id'].values
        
        s1_dicts = [source_map.get(k, {}) for k in s1_ids]
        tgt_dicts = [target_map.get(k, {}) for k in tgt_ids]
        
        s_names = [str(d.get('business_name', '')).strip().lower() for d in s1_dicts]
        t_names = [str(d.get('business_name', '')).strip().lower() for d in tgt_dicts]
        
        s_addrs = [str(d.get('business_address', '')).strip().lower() for d in s1_dicts]
        t_addrs = [str(d.get('business_address', '')).strip().lower() for d in tgt_dicts]
        
        s_countries = [str(d.get('country', '')).strip().lower() for d in s1_dicts]
        t_countries = [str(d.get('country', '')).strip().lower() for d in tgt_dicts]
        
        # 2. Tokenization
        n1_tokens = [set(n.split()) for n in s_names]
        n2_tokens = [set(n.split()) for n in t_names]
        a1_tokens = [set(a.split()) for a in s_addrs]
        a2_tokens = [set(a.split()) for a in t_addrs]
        
        # 3. Vectorized feature generation via list comprehensions (10x faster than iterrows)
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
            
            # Address Features
            addr_overlap = len(a1_tok.intersection(a2_tok))
            addr_jaccard = addr_overlap / len(a1_tok.union(a2_tok)) if a1_tok.union(a2_tok) else 0.0
            addr_len_diff = abs(len(a1) - len(a2))
            
            # Country
            same_country = 1 if c1 == c2 and c1 != "" else 0
            
            features.append([
                s1_ids[i], tgt_ids[i],
                exact_match, token_overlap, jaccard, fuzz_ratio, token_sort, partial, length_diff,
                addr_overlap, addr_jaccard, addr_len_diff,
                same_country
            ])
            
        df_feats = pd.DataFrame(features, columns=[
            'source1_entity_id', 'target_entity_id',
            'exact_match', 'token_overlap', 'jaccard_similarity', 'fuzz_ratio',
            'token_sort_ratio', 'partial_ratio', 'length_difference',
            'address_overlap', 'address_jaccard', 'address_length_difference',
            'same_country'
        ])
        
        # Compute TF-IDF similarities in batch
        if self.is_fitted and len(chunk) > 0:
            vec_n1 = self.tfidf_name.transform(s_names)
            vec_n2 = self.tfidf_name.transform(t_names)
            vec_a1 = self.tfidf_address.transform(s_addrs)
            vec_a2 = self.tfidf_address.transform(t_addrs)
            
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
            logger.info(f"Detected columns for {name}: {list(df.columns)}")
        
        if not self.is_fitted:
            self._fit_tfidf(df_source, df_target)
            
        logger.info("Converting Source/Target to lookup dictionaries...")
        source_map = df_source.set_index('entity_id').to_dict('index')
        target_map = df_target.set_index('entity_id').to_dict('index')
        
        logger.info(f"Generating features in chunks of {self.chunk_size} to save RAM...")
        first_chunk = True
        
        total_chunks = 0
        total_rows = 0
        
        import time
        start_time = time.time()
        
        # Count total rows to give accurate ETA
        total_candidate_rows = 0
        if os.path.exists(candidates_path):
            total_candidate_rows = sum(1 for _ in open(candidates_path)) - 1
            
        for chunk in pd.read_csv(candidates_path, chunksize=self.chunk_size, dtype=str):
            feat_chunk = self._compute_features_chunk(chunk, source_map, target_map)
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            feat_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False
            total_chunks += 1
            total_rows += len(feat_chunk)
            
            elapsed = time.time() - start_time
            rate = total_rows / elapsed * 60 if elapsed > 0 else 0
            
            eta = 0.0
            if total_candidate_rows > 0 and rate > 0:
                remaining = total_candidate_rows - total_rows
                eta = remaining / rate
                
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) if psutil else 0
            logger.info(
                f"[FE STEP] Chunk {total_chunks} | "
                f"Processed: {total_rows}/{total_candidate_rows} | "
                f"Rate: {rate:,.0f} rows/min | "
                f"ETA: {eta:.1f} min | "
                f"RAM: {mem_mb/1024:.1f} GB"
            )
            
        logger.info(f"Feature engineering complete. Total rows processed: {total_rows}")
        # Note: We return empty df here to protect RAM, pipeline will stream output_path for ML.
        return pd.DataFrame()
