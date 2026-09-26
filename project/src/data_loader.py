import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Actual column names from the Amazon ML Challenge dataset
SOURCE_COLUMNS    = ['entity_id', 'business_name', 'business_address', 'country']
GROUND_TRUTH_COLS = ['source1_entity_id', 'source2_entity_id']

class DataLoader:
    def __init__(self, data_dir: str = 'dataset', sep: str = '\t'):
        self.data_dir = Path(data_dir)
        self.sep = sep

    def _load_and_validate(self, filename: str, expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Helper method to load, validate, and report stats for a TSV file."""
        file_path = self.data_dir / filename
        logger.info(f"Loading data from {file_path}...")

        try:
            df = pd.read_csv(file_path, sep=self.sep, dtype=str, encoding='utf-8')
            logger.info(f"Successfully loaded {filename}.")

            if expected_columns:
                self._validate_columns(df, expected_columns, filename)

            self._print_stats(df, filename)
            return df

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Ensure it exists in the {self.data_dir} directory.")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise

    def _validate_columns(self, df: pd.DataFrame, expected_columns: List[str], filename: str) -> None:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns in {filename}: {missing}")
            raise ValueError(f"Missing columns in {filename}: {missing}")
        logger.info(f"Column validation passed for {filename}.")

    def _detect_missing_values(self, df: pd.DataFrame, filename: str) -> None:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logger.warning(f"Missing values detected in {filename}:\n{missing.to_string()}")
        else:
            logger.info(f"No missing values detected in {filename}.")

    def _print_stats(self, df: pd.DataFrame, filename: str) -> None:
        logger.info(f"--- Statistics for {filename} ---")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        self._detect_missing_values(df, filename)
        logger.info("-" * 40)

    def load_source1(self) -> pd.DataFrame:
        """Loads train_source1.tsv — 2,206,821 rows."""
        return self._load_and_validate('train_source1.tsv', SOURCE_COLUMNS)

    def load_source2(self) -> pd.DataFrame:
        """Loads train_source2.tsv — 5,034,616 rows. 168,967 missing addresses."""
        return self._load_and_validate('train_source2.tsv', SOURCE_COLUMNS)

    def load_source3(self) -> pd.DataFrame:
        """Loads train_source3.tsv."""
        return self._load_and_validate('train_source3.tsv', SOURCE_COLUMNS)

    def load_ground_truth(self) -> pd.DataFrame:
        """Loads train_ground_truth.tsv — columns: source1_entity_id, source2_entity_id."""
        return self._load_and_validate('train_ground_truth.tsv')

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads and returns all four datasets."""
        logger.info("Loading all datasets...")
        df1 = self.load_source1()
        df2 = self.load_source2()
        df3 = self.load_source3()
        gt  = self.load_ground_truth()
        logger.info("All datasets loaded successfully.")
        return df1, df2, df3, gt
