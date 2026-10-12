"""
EDA Step 04: Multivariate Analysis
Calculates pairwise correlations among numeric predictors and target variable.
Saves heatmap to reports/figures/eda_correlation_matrix.png.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, save_figure, NUMERIC_FEATURES, TARGET_COL


def run_multivariate_analysis():
    print("=" * 70)
    print(" CREDITFAIR EDA - STEP 04: MULTIVARIATE CORRELATION ANALYSIS")
    print("=" * 70)
    
    df = load_dataset()
    corr_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET_COL]
    corr_matrix = df[corr_cols].corr()
    
    # Top predictors correlated with target
    target_corrs = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print("\n--- Correlations with Target (loan_status) ---")
    print(target_corrs.round(4))
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("CreditFair Feature Correlation Matrix", fontsize=16, weight="bold")
    
    save_figure(fig, "eda_correlation_matrix.png")
    print("Multivariate analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_multivariate_analysis()
