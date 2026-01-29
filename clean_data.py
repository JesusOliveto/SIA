import pandas as pd
import numpy as np
from scipy.io import arff
from pathlib import Path

def clean_data():
    input_path = Path("datasets/winequality.arff")
    output_path = Path("datasets/whinequalityclean.arff")
    
    print(f"Loading {input_path}...")
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    
    # Identify numeric columns (exclude class)
    # meta names are available, or just check types
    # df['class'] is usually bytes in arff load, so likely object/string
    
    print(f"Original shape: {df.shape}")
    
    # Decode bytes if needed (for proper saving later, though we might simpler save as CSV or try to replicate ARFF)
    # The user asked for .arff output. Saving valid ARFF is tricky with pandas alone.
    # We might need to construct it manually or use scipy.io.arff dump if available (it isn't easy).
    # Easier: filter the indices and re-write the ARFF file preserving header, or use liac-arff if installed.
    # Let's try to just filter indices and use the raw data structure if possible, or assume simple text format for ARFF.
    
    # Actually, simpler approach for detection:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude 'class' if it happens to be numeric (it shouldn't be based on previous context, but let's check validation)
    if 'class' in numeric_cols:
        numeric_cols = numeric_cols.drop('class')
        
    print(f"Analyzing features: {list(numeric_cols)}")
    
    # IQR filtering
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # condition: keep rows where ALL features are within bounds? Or ANY?
    # Standard usually: remove if ANY feature is outlier.
    condition = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    
    df_clean = df[condition].copy()
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} rows.")
    
    # Saving as ARFF
    # Since scipy.io.arff doesn't have a 'save', we can write a simple ARFF writer
    # or just copy the header from original and append the cleaned data.
    
    with open(input_path, 'r') as f_in:
        content = f_in.readlines()
        
    # Find @DATA marker
    data_start_idx = 0
    for i, line in enumerate(content):
        if line.strip().upper().startswith("@DATA"):
            data_start_idx = i + 1
            break
            
    header = content[:data_start_idx]
    
    # Prepare data lines
    # We need to format the values exactly as ARFF expects.
    # df values might be decoded.
    # Let's check how loadarff loaded strings (as bytes usually).
    
    with open(output_path, 'w') as f_out:
        f_out.writelines(header)
        # Write rows
        # We need to match the original formatting approx.
        # Since we have the dataframe, we can iterate
        
        for _, row in df_clean.iterrows():
            line_parts = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                if isinstance(val, (int, float)):
                    line_parts.append(str(val))
                else:
                    line_parts.append(str(val))
            f_out.write(",".join(line_parts) + "\n")
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    clean_data()
