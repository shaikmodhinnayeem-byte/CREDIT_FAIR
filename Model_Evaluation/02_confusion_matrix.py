"""
Model Evaluation Step 02: Confusion Matrix
Plots normalized and raw confusion matrix visualization for loan default risk prediction.
Saves figure to reports/figures/confusion_matrix.png.
"""

import sys
from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, save_figure


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "XGBoost Classifier"):
    print("=" * 70)
    print(" CREDITFAIR MODEL EVALUATION - STEP 02: CONFUSION MATRIX")
    print("=" * 70)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Confusion Matrix Audit - {model_name}", fontsize=15, weight="bold")
    
    # 1. Raw Counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Fully Paid (0)", "Default (1)"],
        yticklabels=["Fully Paid (0)", "Default (1)"],
    )
    axes[0].set_title("Absolute Case Volume")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Actual Label")
    
    # 2. Normalized Ratios
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Purples",
        cbar=False,
        ax=axes[1],
        xticklabels=["Fully Paid (0)", "Default (1)"],
        yticklabels=["Fully Paid (0)", "Default (1)"],
    )
    axes[1].set_title("Normalized Classification Rates")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Actual Label")
    
    save_figure(fig, "confusion_matrix.png")
    print(f"Confusion matrix visualized and persisted.")
    print("=" * 70)


if __name__ == "__main__":
    xgb_mod = importlib.import_module("Classification.04_xgboost")
    model, _ = xgb_mod.train_xgboost()
    
    _, X_test, _, y_test, _, _, _ = get_prepared_data()
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
