"""
Model Improvement Step 02: Stratified Cross-Validation
Conducts 5-Fold Stratified Cross-Validation to evaluate model stability,
variance, and absence of fold overfitting.
"""

import sys
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def run_cross_validation():
    print("=" * 70)
    print(" CREDITFAIR MODEL IMPROVEMENT - STEP 02: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    X_train, _, y_train, _, _, _, _ = get_prepared_data()
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
    }
    
    print("[INFO] Running 5-Fold Stratified Cross-Validation across metrics...")
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("\n--- 5-Fold Cross-Validation Summary ---")
    for metric_name, values in scores.items():
        if metric_name.startswith("test_"):
            clean_name = metric_name.replace("test_", "").replace("_", " ").title()
            print(f"  * {clean_name:<20}: Mean = {np.mean(values):.4f} (± {np.std(values):.4f})")
            
    print("=" * 70)
    return scores


if __name__ == "__main__":
    run_cross_validation()
