import pandas as pd
import re
import string
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    DataCleaner handles the normalization of business names, addresses, and countries.
    Production-ready: Uses pre-compiled regexes and optimized pandas operations.
    """
    def __init__(self):
        # Pre-compile regexes for massive performance boost over millions of rows
        self.name_rules = [
            (re.compile(r'\bpvt\b'), 'private'),
            (re.compile(r'\bltd\b'), 'limited'),
            (re.compile(r'\bcorp\b'), 'corporation'),
            (re.compile(r'\bco\b'), 'company')
        ]
        
        self.address_rules = [
            (re.compile(r'\brd\b'), 'road'),
            (re.compile(r'\bst\b'), 'street'),
            (re.compile(r'\bave\b'), 'avenue')
        ]

        # Regex to replace all punctuation with a space (prevents words merging like 'Wal-Mart' -> 'Walmart')
        # We use a custom translation table for blazing fast punctuation removal
        self.punct_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        
        # Regex to collapse multiple spaces
        self.multi_space_re = re.compile(r'\s+')

    def clean_name(self, name: str) -> str:
        if pd.isna(name) or not isinstance(name, str):
            return ""
            
        name = name.lower()
        # Replace punctuation with spaces to preserve word boundaries
        name = name.translate(self.punct_trans)
        
        # Normalize terms using pre-compiled regexes
        for pattern, repl in self.name_rules:
            name = pattern.sub(repl, name)
            
        # Collapse multiple spaces and strip
        return self.multi_space_re.sub(' ', name).strip()

    def clean_address(self, address: str) -> str:
        if pd.isna(address) or not isinstance(address, str):
            return ""
            
        address = address.lower()
        address = address.translate(self.punct_trans)
        
        for pattern, repl in self.address_rules:
            address = pattern.sub(repl, address)
            
        return self.multi_space_re.sub(' ', address).strip()

    def clean_country(self, country: str) -> str:
        if pd.isna(country) or not isinstance(country, str):
            return ""
            
        country = country.lower().strip()
        return self.multi_space_re.sub(' ', country)

    def process(self, df: pd.DataFrame, 
                name_col: str = 'business_name', 
                address_col: str = 'business_address', 
                country_col: str = 'country',
                inplace: bool = False) -> pd.DataFrame:
        """
        Applies cleaning functions to the respective columns of a DataFrame.
        Args:
            inplace: If True, modifies the dataframe in memory to prevent OOM errors on large datasets.
        """
        logger.info("Starting data cleaning process...")
        
        # Prevent memory duplication for large datasets if inplace is requested
        df_clean = df if inplace else df.copy()
        
        # We use map() because it's significantly faster than apply() for Series of strings
        if name_col in df_clean.columns:
            logger.info(f"Cleaning column: {name_col}")
            df_clean[name_col] = df_clean[name_col].map(self.clean_name)
            
        if address_col in df_clean.columns:
            logger.info(f"Cleaning column: {address_col}")
            df_clean[address_col] = df_clean[address_col].map(self.clean_address)
            
        if country_col in df_clean.columns:
            logger.info(f"Cleaning column: {country_col}")
            df_clean[country_col] = df_clean[country_col].map(self.clean_country)
            
        logger.info("Data cleaning completed.")
        return df_clean

# Example usage logic for quick testing
if __name__ == "__main__":
    cleaner = DataCleaner()
    print("Name Examples:")
    print(f"Input: 'Acme Corp., Pvt. Ltd.' -> Output: '{cleaner.clean_name('Acme Corp., Pvt. Ltd.')}'")
    print(f"Input: 'Wal-Mart & Co.' -> Output: '{cleaner.clean_name('Wal-Mart & Co.')}'")
    
    print("\nAddress Examples:")
    print(f"Input: '123 Main St., Apt 4B' -> Output: '{cleaner.clean_address('123 Main St., Apt 4B')}'")
    
    print("\nNull Handling Examples:")
    print(f"Input: None -> Output: '{cleaner.clean_name(None)}'")
