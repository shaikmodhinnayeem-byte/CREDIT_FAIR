"""
Fairness Step 02: Fairness Compliance Metrics
Computes key regulatory quantitative metrics:
1. Disparate Impact Ratio (80% / Four-Fifths Rule compliance)
2. Statistical Parity Difference (Demographic Parity)
3. Equal Opportunity Difference (True Positive Rate Parity)
4. Average Odds Difference
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, load_artifact, MODEL_PATH


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_approved: np.ndarray,
    group_series: pd.Series,
    privileged_group: str = "MORTGAGE",
    unprivileged_group: str = "RENT",
) -> Dict[str, float]:
    """Calculates industry standard fair lending compliance metrics."""
    priv_mask = (group_series == privileged_group).values
    unpriv_mask = (group_series == unprivileged_group).values
    
    # Selection (Approval) Rates
    priv_approval_rate = np.mean(y_approved[priv_mask])
    unpriv_approval_rate = np.mean(y_approved[unpriv_mask])
    
    # 1. Disparate Impact Ratio (DIR)
    dir_ratio = (unpriv_approval_rate / priv_approval_rate) if priv_approval_rate > 0 else 1.0
    
    # 2. Statistical Parity Difference (SPD)
    spd = unpriv_approval_rate - priv_approval_rate
    
    # Good credit borrowers (y_true == 0, non-defaulters)
    good_credit = (y_true == 0)
    priv_tpr = np.mean(y_approved[priv_mask & good_credit]) if np.sum(priv_mask & good_credit) > 0 else 0.0
    unpriv_tpr = np.mean(y_approved[unpriv_mask & good_credit]) if np.sum(unpriv_mask & good_credit) > 0 else 0.0
    
    # 3. Equal Opportunity Difference (EOD)
    eod = unpriv_tpr - priv_tpr
    
    # Bad credit borrowers (y_true == 1, defaulters erroneously approved = False Positive in approval)
    bad_credit = (y_true == 1)
    priv_fpr = np.mean(y_approved[priv_mask & bad_credit]) if np.sum(priv_mask & bad_credit) > 0 else 0.0
    unpriv_fpr = np.mean(y_approved[unpriv_mask & bad_credit]) if np.sum(unpriv_mask & bad_credit) > 0 else 0.0
    
    # 4. Average Odds Difference (AOD)
    aod = 0.5 * ((unpriv_fpr - priv_fpr) + (unpriv_tpr - priv_tpr))
    
    metrics = {
        f"Privileged Group ({privileged_group}) Approval Rate": priv_approval_rate,
        f"Unprivileged Group ({unprivileged_group}) Approval Rate": unpriv_approval_rate,
        "Disparate Impact Ratio (DIR)": dir_ratio,
        "Statistical Parity Difference": spd,
        "Equal Opportunity Difference": eod,
        "Average Odds Difference": aod,
    }
    return metrics


def run_fairness_metrics():
    print("=" * 70)
    print(" CREDITFAIR FAIRNESS - STEP 02: FAIRNESS METRICS AUDIT")
    print("=" * 70)
    
    _, X_test, _, y_test, _, _, test_df = get_prepared_data()
    model = load_artifact(MODEL_PATH)
    probs = model.predict_proba(X_test)[:, 1]
    approved = (probs <= 0.25).astype(int)
    
    metrics = compute_fairness_metrics(
        y_true=y_test,
        y_approved=approved,
        group_series=test_df["home_ownership"],
        privileged_group="MORTGAGE",
        unprivileged_group="RENT",
    )
    
    print("\n--- Compliance Audit: MORTGAGE (Privileged) vs RENT (Unprivileged) ---")
    for k, v in metrics.items():
        print(f"  * {k:<45}: {v:.4f}")
        
    dir_val = metrics["Disparate Impact Ratio (DIR)"]
    is_compliant = dir_val >= 0.80
    print(f"\n[REGULATORY VERDICT] 80% Rule (Four-Fifths) Compliant: {'PASS (DIR >= 0.80)' if is_compliant else 'FLAGGED'}")
    print("=" * 70)
    return metrics


if __name__ == "__main__":
    run_fairness_metrics()
