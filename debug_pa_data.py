import pandas as pd
import numpy as np

# Load data
data_pa_non = pd.read_csv('Pa_renal_nontransplanted.csv')
data_pa_post = pd.read_csv('pa_renal_post-transplanted.csv')

print("Original data shapes:")
print(f"Non-transplanted: {data_pa_non.shape}")
print(f"Post-transplant: {data_pa_post.shape}")

# Check for header rows in data
print("\nFirst few rows of non-transplanted:")
print(data_pa_non.head(3))

print("\nFirst few rows of post-transplant:")
print(data_pa_post.head(3))

# Check DOS column for issues
print("\nDOS column values in non-transplanted:")
print(data_pa_non['DOS'].head(10))

print("\nDOS column values in post-transplant:")
print(data_pa_post['DOS'].head(10))

# Remove rows where DOS contains the header
data_pa_non_clean = data_pa_non[~data_pa_non['DOS'].astype(str).str.contains('DOS', na=False)]
data_pa_post_clean = data_pa_post[~data_pa_post['DOS'].astype(str).str.contains('DOS', na=False)]

print("\nAfter removing header rows:")
print(f"Non-transplanted: {data_pa_non_clean.shape}")
print(f"Post-transplant: {data_pa_post_clean.shape}")

# Standardize column names
COLUMN_ALIASES = {
    'Arg_Citrulline_Ratio': 'Arginine_Citrulline_ratio',
    'eGFR_CyC': 'eGFR_CystatinC',
    'Screatinine_mg_dL': 'Creatinine',
    'Cystatin_mg_L': 'Cystatin_C',
}

data_pa_non_clean = data_pa_non_clean.rename(columns=COLUMN_ALIASES)
data_pa_post_clean = data_pa_post_clean.rename(columns=COLUMN_ALIASES)

# Combine
data_all = pd.concat([data_pa_non_clean, data_pa_post_clean], ignore_index=True)
print(f"\nCombined dataset: {data_all.shape}")

# Check key columns for data types
key_cols = ['Protein_Creatinine_Ratio', 'eGFR_SCr', 'LN_TNFR1', 'Uric_Acid']
for col in key_cols:
    if col in data_all.columns:
        print(f"\n{col}:")
        print(f"  dtype: {data_all[col].dtype}")
        print(f"  sample values: {data_all[col].head(5).tolist()}")
        print(f"  non-null count: {data_all[col].notna().sum()}")
        
        # Try to convert to numeric
        numeric_col = pd.to_numeric(data_all[col], errors='coerce')
        print(f"  after numeric conversion: {numeric_col.notna().sum()} non-null")
        
        # Check for any remaining string values
        if data_all[col].dtype == 'object':
            string_vals = data_all[col][data_all[col].astype(str).str.contains(r'[a-zA-Z]', na=False)]
            if len(string_vals) > 0:
                print(f"  WARNING: Found {len(string_vals)} string values")
                print(f"  Sample string values: {string_vals.head(3).tolist()}")

print("\nData cleaning complete")
