"""
Fairness Step 03: Fairness Visualization
Visualizes loan approval rates and Disparate Impact ratios across demographic groups.
Saves figure to reports/figures/fairness_visualization.png.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data, load_artifact, save_figure, MODEL_PATH


def plot_fairness_audit(threshold: float = 0.25):
    print("=" * 70)
    print(" CREDITFAIR FAIRNESS - STEP 03: FAIRNESS VISUALIZATION")
    print("=" * 70)
    
    _, X_test, _, y_test, _, _, test_df = get_prepared_data()
    model = load_artifact(MODEL_PATH)
    probs = model.predict_proba(X_test)[:, 1]
    
    test_eval = test_df.copy()
    test_eval["loan_approved"] = (probs <= threshold).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CreditFair Fair Lending Compliance Audit", fontsize=15, weight="bold")
    
    # 1. Approval Rate by Home Ownership
    home_appr = test_eval.groupby("home_ownership")["loan_approved"].mean().reset_index()
    sns.barplot(
        data=home_appr,
        x="home_ownership",
        y="loan_approved",
        hue="home_ownership",
        legend=False,
        palette="Blues_d",
        ax=axes[0],
    )
    axes[0].set_title("Approval Rate by Home Ownership Subgroup")
    axes[0].set_ylabel("Underwriting Approval Rate")
    axes[0].set_xlabel("Home Ownership")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', va='bottom', weight='bold')
        
    # 2. Approval Rate by Income Bracket
    inc_appr = test_eval.groupby("income_bracket")["loan_approved"].mean().reset_index()
    sns.barplot(
        data=inc_appr,
        x="income_bracket",
        y="loan_approved",
        hue="income_bracket",
        legend=False,
        palette="Greens_d",
        ax=axes[1],
    )
    axes[1].set_title("Approval Rate by Income Tier")
    axes[1].set_ylabel("Underwriting Approval Rate")
    axes[1].set_xlabel("Income Tier")
    axes[1].set_ylim(0, 1.0)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', va='bottom', weight='bold')
    
    save_figure(fig, "fairness_visualization.png")
    print(f"Fairness audit visualization saved.")
    print("=" * 70)


if __name__ == "__main__":
    plot_fairness_audit()
