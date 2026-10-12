"""
Model Evaluation Step 03: ROC & PR Curve Analysis
Renders and saves Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
Saves figure to reports/figures/roc_curve.png.
"""

import sys
from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, save_figure


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "XGBoost Classifier"):
    print("=" * 70)
    print(" CREDITFAIR MODEL EVALUATION - STEP 03: ROC & PR CURVES")
    print("=" * 70)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline_pr = np.mean(y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Discrimination & Precision-Recall Dynamics - {model_name}", fontsize=15, weight="bold")
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. PR Curve
    axes[1].plot(recall, precision, color="#8e44ad", lw=2.5, label=f"PR Curve (AP = {pr_auc:.4f})")
    axes[1].axhline(y=baseline_pr, color="#e74c3c", linestyle="--", label=f"Baseline Prevalence ({baseline_pr:.2%})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Coverage)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    save_figure(fig, "roc_curve.png")
    print(f"ROC and PR curves saved.")
    print("=" * 70)


if __name__ == "__main__":
    xgb_mod = importlib.import_module("Classification.04_xgboost")
    model, _ = xgb_mod.train_xgboost()
    
    _, X_test, _, y_test, _, _, _ = get_prepared_data()
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_roc_pr_curves(y_test, y_prob)
