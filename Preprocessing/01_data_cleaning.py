"""
Preprocessing Step 01: Data Cleaning
Handles string conversions, numeric parsing, and imputation of missing values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, TARGET_COL


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs rigorous missing value imputation and data integrity sanitation."""
    df_clean = df.copy()

    # Numeric imputations using median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Categorical imputations using mode
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Remove any unneeded string identifiers if present
    drop_candidates = ["id", "member_id", "url"]
    existing_drops = [c for c in drop_candidates if c in df_clean.columns]
    if existing_drops:
        df_clean = df_clean.drop(columns=existing_drops)

    return df_clean


def run_data_cleaning():
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 01: DATA CLEANING")
    print("=" * 70)
    
    raw_df = load_dataset()
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Total missing values before cleaning: {raw_df.isnull().sum().sum()}")
    
    cleaned_df = clean_credit_data(raw_df)
    print(f"Total missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
    print("Data cleaning successfully completed.")
    print("=" * 70)
    return cleaned_df


if __name__ == "__main__":
    run_data_cleaning()
