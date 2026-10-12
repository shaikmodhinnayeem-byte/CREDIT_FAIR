"""
Preprocessing Step 02: Feature Engineering
Constructs financial debt-to-burden ratios, line utilization thresholds,
and derogatory risk indicator flags.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, engineer_credit_features, TARGET_COL
import importlib
data_clean_mod = importlib.import_module("Preprocessing.01_data_cleaning")
clean_credit_data = data_clean_mod.clean_credit_data


def run_feature_engineering():
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 02: FEATURE ENGINEERING")
    print("=" * 70)
    
    raw_df = load_dataset()
    cleaned_df = clean_credit_data(raw_df)
    engineered_df = engineer_credit_features(cleaned_df)
    
    new_cols = [
        "installment_to_income",
        "high_utilization_flag",
        "open_to_total_acc_ratio",
        "has_derogatory_record",
    ]
    
    print("\n--- Summary of Engineered Financial Risk Features ---")
    print(engineered_df[new_cols].describe().round(4))
    
    print("\n--- Correlation with Target (loan_status) ---")
    for col in new_cols:
        corr = engineered_df[col].corr(engineered_df[TARGET_COL])
        print(f"  * {col:<26}: Pearson r = {corr:+.4f}")
        
    print(f"\nFinal Feature Count: {engineered_df.shape[1]} features")
    print("Feature engineering successfully completed.")
    print("=" * 70)
    return engineered_df


if __name__ == "__main__":
    run_feature_engineering()
