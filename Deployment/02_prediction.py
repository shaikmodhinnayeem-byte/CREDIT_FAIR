"""
Deployment Step 02: Real-Time Loan Applicant Prediction Service
Receives raw applicant data payload, applies preprocessing, evaluates default risk,
and returns automated underwriting decision and risk band.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import importlib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import engineer_credit_features

loader_mod = importlib.import_module("Deployment.01_load_model")
load_production_pipeline = loader_mod.load_production_pipeline


def predict_applicant(applicant: Dict[str, Any], threshold: float = 0.25) -> Dict[str, Any]:
    """
    Evaluates loan applicant risk and makes an underwriting determination.
    
    Parameters:
        applicant: Dictionary containing raw applicant features.
        threshold: Max allowed predicted default probability for approval.
        
    Returns:
        Structured dictionary with risk score, default probability, decision, and risk tier.
    """
    preprocessor, model = load_production_pipeline()
    
    # Format input into single-row DataFrame
    df_raw = pd.DataFrame([applicant])
    df_features = engineer_credit_features(df_raw)
    
    # Transform using serialized preprocessor
    X_proc = preprocessor.transform(df_features)
    
    # Predict default probability
    default_prob = float(model.predict_proba(X_proc)[0, 1])
    approved = bool(default_prob <= threshold)
    
    # Assign Risk Tier
    if default_prob < 0.15:
        risk_tier = "Prime / Low Risk"
    elif default_prob < 0.30:
        risk_tier = "Near-Prime / Moderate Risk"
    else:
        risk_tier = "Subprime / High Risk"
        
    return {
        "status": "success",
        "loan_decision": "APPROVED" if approved else "DECLINED",
        "default_probability": round(default_prob, 4),
        "default_probability_percent": f"{default_prob:.1%}",
        "decision_threshold": threshold,
        "risk_tier": risk_tier,
        "applicant_profile": {
            "loan_amount": applicant.get("loan_amnt"),
            "annual_income": applicant.get("annual_inc"),
            "fico_score": applicant.get("fico_score"),
            "dti": applicant.get("dti"),
            "home_ownership": applicant.get("home_ownership"),
        }
    }


def run_sample_predictions():
    print("=" * 70)
    print(" CREDITFAIR DEPLOYMENT - STEP 02: INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    # Sample Applicant 1: Strong credit profile
    applicant_a = {
        "loan_amnt": 12000.0,
        "term_months": 36,
        "int_rate": 7.5,
        "installment": 373.2,
        "annual_inc": 95000.0,
        "dti": 11.2,
        "fico_score": 765.0,
        "open_acc": 12,
        "pub_rec": 0,
        "revol_bal": 4500.0,
        "revol_util": 18.0,
        "total_acc": 28,
        "mort_acc": 2,
        "pub_rec_bankruptcies": 0,
        "emp_length_years": 8.0,
        "home_ownership": "MORTGAGE",
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "income_bracket": "High",
    }
    
    # Sample Applicant 2: Elevated risk profile
    applicant_b = {
        "loan_amnt": 28000.0,
        "term_months": 60,
        "int_rate": 22.4,
        "installment": 780.0,
        "annual_inc": 38000.0,
        "dti": 36.5,
        "fico_score": 635.0,
        "open_acc": 8,
        "pub_rec": 1,
        "revol_bal": 18500.0,
        "revol_util": 89.0,
        "total_acc": 14,
        "mort_acc": 0,
        "pub_rec_bankruptcies": 1,
        "emp_length_years": 1.0,
        "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "purpose": "small_business",
        "income_bracket": "Low",
    }
    
    print("\n--- Evaluating Applicant A (Prime Profile) ---")
    res_a = predict_applicant(applicant_a)
    for k, v in res_a.items():
        if k != "applicant_profile":
            print(f"  * {k:<28}: {v}")
            
    print("\n--- Evaluating Applicant B (High Risk Profile) ---")
    res_b = predict_applicant(applicant_b)
    for k, v in res_b.items():
        if k != "applicant_profile":
            print(f"  * {k:<28}: {v}")
            
    print("=" * 70)


if __name__ == "__main__":
    run_sample_predictions()
