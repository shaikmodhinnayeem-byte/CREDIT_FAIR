"""
EDA Step 01: Data Understanding
Inspects shapes, data types, missing value ratios, and target class balance.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, TARGET_COL


def run_data_understanding():
    print("=" * 70)
    print(" CREDITFAIR EDA - STEP 01: DATA UNDERSTANDING")
    print("=" * 70)
    
    df = load_dataset()
    print(f"\n[INFO] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n--- Feature Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values Audit ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct.round(2)})
    missing_summary = missing_df[missing_df["Missing Count"] > 0]
    if not missing_summary.empty:
        print(missing_summary)
    else:
        print("No missing values detected.")
        
    print("\n--- Target Variable Distribution (loan_status) ---")
    target_counts = df[TARGET_COL].value_counts()
    target_norm = df[TARGET_COL].value_counts(normalize=True) * 100
    target_df = pd.DataFrame({"Count": target_counts, "Percentage (%)": target_norm.round(2)})
    print(target_df)
    
    print("\n--- Five-Point Numeric Summary ---")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(2))
    
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_data_understanding()
