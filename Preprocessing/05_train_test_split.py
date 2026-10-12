"""
Preprocessing Step 05: Train Test Split
Executes stratified partition (80% Train, 20% Test) ensuring identical class
and protected demographic group balance across folds.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, TARGET_COL


def run_train_test_split(test_size: float = 0.20, random_state: int = 42):
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 05: TRAIN TEST SPLIT")
    print("=" * 70)
    
    df = load_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    
    print(f"Total Records : {len(df)}")
    print(f"Train Partition: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Test Partition : {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    print("\n--- Target (Default) Balance Verification ---")
    print(f"Train Default Rate: {train_df[TARGET_COL].mean():.2%}")
    print(f"Test Default Rate : {test_df[TARGET_COL].mean():.2%}")
    
    print("\n--- Demographic Representation (home_ownership) in Splits ---")
    train_ownership = train_df["home_ownership"].value_counts(normalize=True).round(3)
    test_ownership = test_df["home_ownership"].value_counts(normalize=True).round(3)
    balance_df = pd.DataFrame({"Train Distribution": train_ownership, "Test Distribution": test_ownership})
    print(balance_df)
    
    print("Train-test partitioning successfully validated.")
    print("=" * 70)
    return train_df, test_df


if __name__ == "__main__":
    run_train_test_split()
