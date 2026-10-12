"""
Model Improvement Step 01: Hyperparameter Tuning
Performs RandomizedSearchCV on XGBoost to discover optimal regularization,
tree depth, and learning rates.
"""

import sys
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def run_hyperparameter_tuning():
    print("=" * 70)
    print(" CREDITFAIR MODEL IMPROVEMENT - STEP 01: HYPERPARAMETER TUNING")
    print("=" * 70)
    
    X_train, _, y_train, _, _, _, _ = get_prepared_data()
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    base_xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    
    param_dist = {
        "n_estimators": [80, 120, 150],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }
    
    print("[INFO] Launching RandomizedSearchCV (n_iter=8, cv=3, scoring=roc_auc)...")
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=8,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    
    print("\n--- Hyperparameter Tuning Findings ---")
    print(f"Best ROC-AUC Score: {search.best_score_:.4f}")
    print("Optimal Hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  * {k:<20}: {v}")
        
    print("=" * 70)
    return search.best_params_, search.best_estimator_


if __name__ == "__main__":
    run_hyperparameter_tuning()
