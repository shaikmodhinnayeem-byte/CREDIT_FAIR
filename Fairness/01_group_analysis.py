"""
Fairness Step 01: Subgroup Disparity Analysis
Analyzes model predictions, approval rates, and actual defaults across protected
and socio-economic demographic subgroups (home ownership and income bracket).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, load_artifact, MODEL_PATH


def run_group_analysis(threshold: float = 0.25):
    print("=" * 70)
    print(" CREDITFAIR FAIRNESS - STEP 01: SUBGROUP DISPARITY ANALYSIS")
    print("=" * 70)
    
    _, X_test, _, y_test, _, _, test_df = get_prepared_data()
    
    # Load model
    model = load_artifact(MODEL_PATH)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Underwriting approval decision: Approved if default probability <= threshold
    test_eval = test_df.copy()
    test_eval["pred_default_prob"] = probs
    test_eval["loan_approved"] = (probs <= threshold).astype(int)
    test_eval["actual_default"] = y_test
    
    print(f"[INFO] Underwriting Default Probability Cutoff Threshold: {threshold:.2f}")
    
    # 1. Analysis across Home Ownership
    print("\n--- Subgroup Audit by Home Ownership ---")
    group_home = test_eval.groupby("home_ownership").agg(
        Total_Applicants=("loan_approved", "count"),
        Approval_Rate=("loan_approved", "mean"),
        Actual_Default_Rate=("actual_default", "mean"),
        Mean_Default_Probability=("pred_default_prob", "mean"),
    ).round(4)
    print(group_home)
    
    # 2. Analysis across Income Bracket
    print("\n--- Subgroup Audit by Income Bracket ---")
    group_inc = test_eval.groupby("income_bracket").agg(
        Total_Applicants=("loan_approved", "count"),
        Approval_Rate=("loan_approved", "mean"),
        Actual_Default_Rate=("actual_default", "mean"),
        Mean_Default_Probability=("pred_default_prob", "mean"),
    ).round(4)
    print(group_inc)
    
    print("=" * 70)
    return test_eval


if __name__ == "__main__":
    run_group_analysis()
