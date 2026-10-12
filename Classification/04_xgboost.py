"""
Classification Model 04: XGBoost
Trains and evaluates a high-performance Extreme Gradient Boosting (XGBoost) classifier.
"""

import sys
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def train_xgboost():
    print("=" * 70)
    print(" CREDITFAIR CLASSIFICATION - MODEL 04: XGBOOST")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, _, _ = get_prepared_data()
    
    # Calculate scale_pos_weight
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    
    print("[INFO] Training XGBoost Classifier...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    
    print("\n--- Test Partition Performance ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  * {k:<15}: {v:.4f}")
            
    print("=" * 70)
    return model, metrics


if __name__ == "__main__":
    train_xgboost()
