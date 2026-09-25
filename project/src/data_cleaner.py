import pandas as pd
import re
import string
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    DataCleaner handles the normalization of business names, addresses, and countries.
    """
    def __init__(self):
        # Normalization mappings based on requirements
        self.name_normalizations = {
            r'\bpvt\b': 'private',
            r'\bltd\b': 'limited',
            r'\bcorp\b': 'corporation',
            r'\bco\b': 'company'
        }
        
        self.address_normalizations = {
            r'\brd\b': 'road',
            r'\bst\b': 'street',
            r'\bave\b': 'avenue'
        }

    def _apply_rules(self, text: str, rules: dict) -> str:
        """Applies regex pattern replacements sequentially."""
        for pattern, replacement in rules.items():
            text = re.sub(pattern, replacement, text)
        return text

    def clean_name(self, name: str) -> str:
        """
        Cleans a business name.
        - Handles null values
        - Lowercases
        - Removes punctuation
        - Normalizes specific terms (pvt, ltd, corp, co)
        - Removes extra spaces
        
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_name("Acme Corp., Pvt. Ltd.")
            'acme corporation private limited'
            >>> cleaner.clean_name(None)
            ''
            >>> cleaner.clean_name("  Google   Co ")
            'google company'
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
            
        name = name.lower()
        
        # Remove punctuation
        name = name.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize terms
        name = self._apply_rules(name, self.name_normalizations)
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    def clean_address(self, address: str) -> str:
        """
        Cleans a business address.
        - Handles null values
        - Lowercases
        - Removes punctuation
        - Normalizes specific terms (rd, st, ave)
        - Removes extra spaces
        
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_address("123 Main St., Apt 4B")
            '123 main street apt 4b'
            >>> cleaner.clean_address(float('nan'))
            ''
            >>> cleaner.clean_address("5th Ave. & 42nd Rd.")
            '5th avenue 42nd road'
        """
        if pd.isna(address) or not isinstance(address, str):
            return ""
            
        address = address.lower()
        
        # Remove punctuation
        address = address.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize terms
        address = self._apply_rules(address, self.address_normalizations)
        
        # Remove extra spaces
        address = re.sub(r'\s+', ' ', address).strip()
        
        return address

    def clean_country(self, country: str) -> str:
        """
        Cleans a country name.
        - Handles null values
        - Lowercases
        - Removes extra spaces
        
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_country("  United States  ")
            'united states'
            >>> cleaner.clean_country(None)
            ''
        """
        if pd.isna(country) or not isinstance(country, str):
            return ""
            
        country = country.lower().strip()
        country = re.sub(r'\s+', ' ', country)
        
        return country

    def process(self, df: pd.DataFrame, 
                name_col: str = 'business_name', 
                address_col: str = 'business_address', 
                country_col: str = 'country') -> pd.DataFrame:
        """
        Applies cleaning functions to the respective columns of a DataFrame.
        """
        logger.info("Starting data cleaning process...")
        df_clean = df.copy()
        
        if name_col in df_clean.columns:
            logger.info(f"Cleaning column: {name_col}")
            df_clean[name_col] = df_clean[name_col].apply(self.clean_name)
            
        if address_col in df_clean.columns:
            logger.info(f"Cleaning column: {address_col}")
            df_clean[address_col] = df_clean[address_col].apply(self.clean_address)
            
        if country_col in df_clean.columns:
            logger.info(f"Cleaning column: {country_col}")
            df_clean[country_col] = df_clean[country_col].apply(self.clean_country)
            
        logger.info("Data cleaning completed.")
        return df_clean

# Example usage logic for quick testing
if __name__ == "__main__":
    cleaner = DataCleaner()
    print("Name Examples:")
    print(f"Input: 'Acme Corp., Pvt. Ltd.' -> Output: '{cleaner.clean_name('Acme Corp., Pvt. Ltd.')}'")
    print(f"Input: '  Google   Co ' -> Output: '{cleaner.clean_name('  Google   Co ')}'")
    
    print("\nAddress Examples:")
    print(f"Input: '123 Main St., Apt 4B' -> Output: '{cleaner.clean_address('123 Main St., Apt 4B')}'")
    print(f"Input: '5th Ave. & 42nd Rd.' -> Output: '{cleaner.clean_address('5th Ave. & 42nd Rd.')}'")
    
    print("\nNull Handling Examples:")
    print(f"Input: None -> Output: '{cleaner.clean_name(None)}'")
    print(f"Input: float('nan') -> Output: '{cleaner.clean_address(float('nan'))}'")
