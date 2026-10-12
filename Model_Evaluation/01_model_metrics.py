"""
Model Evaluation Step 01: Model Metrics
Computes comprehensive diagnostic metrics: Accuracy, Balanced Accuracy,
Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, and Brier Score.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates full enterprise suite of classification performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (PPV)": precision_score(y_true, y_pred, zero_division=0),
        "Recall / Sensitivity (TPR)": recall_score(y_true, y_pred),
        "Specificity (TNR)": specificity,
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC (Avg Precision)": average_precision_score(y_true, y_prob),
        "Brier Calibration Loss": brier_score_loss(y_true, y_prob),
    }
    return metrics


def print_metrics_table(metrics: Dict[str, float], title: str = "Model Performance Report") -> None:
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  * {k:<30}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    from project_utils import get_prepared_data
    import importlib
    xgb_mod = importlib.import_module("Classification.04_xgboost")
    model, _ = xgb_mod.train_xgboost()
    
    _, X_test, _, y_test, _, _, _ = get_prepared_data()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    res = compute_comprehensive_metrics(y_test, y_pred, y_prob)
    print_metrics_table(res, "XGBoost Production Evaluation")
