import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.io import arff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_arff(path: Path) -> pd.DataFrame:
    """Loads an ARFF file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading data from {path}...")
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        
        # Decode bytes columns to strings if necessary
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                except Exception as e:
                    logger.warning(f"Could not decode column {col}: {e}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load ARFF file: {e}")
        raise

def clean_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers using the IQR method.
    
    Args:
        df: Input DataFrame.
        factor: IQR multiplier (default 1.5).
        
    Returns:
        Cleaned DataFrame.
    """
    logger.info("Detecting outliers using IQR method...")
    
    # Select only numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude 'class' column if it exists and is numeric (it shouldn't be touched)
    if 'class' in numeric_cols:
        numeric_cols = numeric_cols.drop('class')
        
    logger.info(f"Analyzing features: {list(numeric_cols)}")
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Keep rows where ALL features are within bounds
    # Alternatively: Remove row if ANY feature is an outlier
    condition = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    
    df_clean = df[condition].copy()
    
    removed_count = len(df) - len(df_clean)
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Cleaned shape: {df_clean.shape}")
    logger.info(f"Removed {removed_count} rows ({removed_count/len(df):.2%}).")
    
    return df_clean

def save_arff(df: pd.DataFrame, input_path: Path, output_path: Path) -> None:
    """
    Saves the DataFrame as an ARFF file, preserving the header from the input file.
    
    Args:
        df: DataFrame to save.
        input_path: Path to the original ARFF file (source of header).
        output_path: Path to write the new ARFF file.
    """
    logger.info(f"Saving cleaned data to {output_path}...")
    
    try:
        with open(input_path, 'r') as f_in:
            content = f_in.readlines()
        
        # Locate @DATA tag
        data_start_idx = 0
        for i, line in enumerate(content):
            if line.strip().upper().startswith("@DATA"):
                data_start_idx = i + 1
                break
        
        header = content[:data_start_idx]
        
        with open(output_path, 'w') as f_out:
            f_out.writelines(header)
            
            # Write data rows
            # We iterate to ensure correct formatting (strings vs numbers)
            for _, row in df.iterrows():
                line_parts = []
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        line_parts.append(str(val))
                    else:
                        line_parts.append(str(val))
                f_out.write(",".join(line_parts) + "\n")
                
        logger.info("Save successful.")
        
    except Exception as e:
        logger.error(f"Failed to save ARFF file: {e}")
        raise

def main():
    base_dir = Path("datasets")
    input_file = base_dir / "winequality.arff"
    output_file = base_dir / "whinequalityclean.arff" # Keeping typo 'whine' to maintain compatibility with app.py
    
    # Correcting typo for clarity if possible, but app.py expects 'whinequalityclean.arff'.
    # I will stick to what app.py expects unless I change app.py too.
    # Let's verify if app.py has the typo or if it was just in this script.
    # app.py line 23: def load_data(filename: str = "whinequalityclean.arff")
    # Yes, app.py has the typo. I will keep it for now to avoid breaking changes, 
    # or I should fix it in both places. I'll check if I can fix it later.
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    try:
        df = load_arff(input_file)
        df_clean = clean_outliers(df)
        save_arff(df_clean, input_file, output_file)
        logger.info("Data cleaning process completed.")
    except Exception:
        logger.error("Process failed.")

if __name__ == "__main__":
    main()
