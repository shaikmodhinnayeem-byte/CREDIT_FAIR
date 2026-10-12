"""
Model Improvement Step 03: Final Improved Model with Probability Calibration
Trains the enhanced tuned XGBoost classifier and applies Platt Scaling
(probability calibration) for reliable underwriting risk scoring.
Serializes final champion model to models/creditfair_model.pkl.
"""

import sys
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, save_artifact, MODEL_PATH


def train_improved_model():
    print("=" * 70)
    print(" CREDITFAIR MODEL IMPROVEMENT - STEP 03: IMPROVED & CALIBRATED MODEL")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, _, _ = get_prepared_data()
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    base_xgb = XGBClassifier(
        n_estimators=140,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    
    print("[INFO] Fitting CalibratedClassifierCV (Platt Scaling) over XGBoost with 12 parallel threads...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )
    calibrated_model.fit(X_train, y_train)
    
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Brier Score": brier_score_loss(y_test, y_prob),
    }
    
    print("\n--- Calibrated Champion Model Metrics on Holdout Test ---")
    for k, v in metrics.items():
        print(f"  * {k:<15}: {v:.4f}")
        
    # Serialize production artifact
    save_artifact(calibrated_model, MODEL_PATH)
    print(f"\n[SUCCESS] Final CreditFair champion model serialized to: {MODEL_PATH}")
    print("=" * 70)
    return calibrated_model, metrics


if __name__ == "__main__":
    train_improved_model()
