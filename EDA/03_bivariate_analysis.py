"""
EDA Step 03: Bivariate Analysis
Explores relationships between candidate predictive features, protected attributes,
and loan default status.
Saves figure to reports/figures/eda_bivariate.png.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, save_figure, TARGET_COL


def run_bivariate_analysis():
    print("=" * 70)
    print(" CREDITFAIR EDA - STEP 03: BIVARIATE ANALYSIS")
    print("=" * 70)
    
    df = load_dataset()
    sns.set_theme(style="whitegrid", palette="deep")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("CreditFair Bivariate Risk Analysis", fontsize=16, weight="bold")
    
    # 1. FICO Score vs Default Status
    sns.boxplot(ax=axes[0, 0], x=TARGET_COL, y="fico_score", data=df, hue=TARGET_COL, legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[0, 0].set_title("FICO Score by Loan Outcome")
    axes[0, 0].set_xlabel("Loan Outcome (0: Paid, 1: Default)")
    axes[0, 0].set_ylabel("FICO Score")
    
    # 2. Interest Rate vs Default Status
    sns.boxplot(ax=axes[0, 1], x=TARGET_COL, y="int_rate", data=df, hue=TARGET_COL, legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[0, 1].set_title("Interest Rate by Loan Outcome")
    axes[0, 1].set_xlabel("Loan Outcome (0: Paid, 1: Default)")
    axes[0, 1].set_ylabel("Interest Rate (%)")
    
    # 3. Default Rate across Home Ownership (Demographic / Fairness Proxy)
    ownership_default = df.groupby("home_ownership")[TARGET_COL].mean().reset_index()
    sns.barplot(ax=axes[1, 0], x="home_ownership", y=TARGET_COL, data=ownership_default, hue="home_ownership", legend=False, palette="Blues_r")
    axes[1, 0].set_title("Default Rate by Home Ownership")
    axes[1, 0].set_xlabel("Home Ownership")
    axes[1, 0].set_ylabel("Default Rate")
    axes[1, 0].tick_params(axis="x", rotation=25)
    
    # 4. Default Rate across Income Bracket
    income_default = df.groupby("income_bracket")[TARGET_COL].mean().reset_index()
    sns.barplot(ax=axes[1, 1], x="income_bracket", y=TARGET_COL, data=income_default, hue="income_bracket", legend=False, palette="Purples_r")
    axes[1, 1].set_title("Default Rate by Income Bracket")
    axes[1, 1].set_xlabel("Income Bracket")
    axes[1, 1].set_ylabel("Default Rate")
    
    save_figure(fig, "eda_bivariate.png")
    print("Bivariate analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_bivariate_analysis()
