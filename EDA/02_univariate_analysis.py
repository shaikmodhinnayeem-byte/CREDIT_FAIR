"""
EDA Step 02: Univariate Analysis
Analyzes and visualizes individual distributions of key numerical and categorical variables.
Saves figure to reports/figures/eda_univariate.png.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, save_figure, TARGET_COL


def run_univariate_analysis():
    print("=" * 70)
    print(" CREDITFAIR EDA - STEP 02: UNIVARIATE ANALYSIS")
    print("=" * 70)
    
    df = load_dataset()
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CreditFair Univariate Distributions", fontsize=16, weight="bold")
    
    # 1. Target Distribution
    sns.countplot(ax=axes[0, 0], x=TARGET_COL, data=df, hue=TARGET_COL, legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[0, 0].set_title("Target Distribution (0: Fully Paid, 1: Default)")
    axes[0, 0].set_xlabel("Loan Status")
    axes[0, 0].set_ylabel("Count")
    
    # 2. Loan Amount
    sns.histplot(ax=axes[0, 1], data=df, x="loan_amnt", kde=True, color="#3498db", bins=30)
    axes[0, 1].set_title("Loan Amount Distribution")
    axes[0, 1].set_xlabel("Loan Amount ($)")
    
    # 3. Interest Rate
    sns.histplot(ax=axes[0, 2], data=df, x="int_rate", kde=True, color="#9b59b6", bins=30)
    axes[0, 2].set_title("Interest Rate (%) Distribution")
    axes[0, 2].set_xlabel("Interest Rate (%)")
    
    # 4. FICO Score
    sns.histplot(ax=axes[1, 0], data=df, x="fico_score", kde=True, color="#e67e22", bins=30)
    axes[1, 0].set_title("FICO Credit Score Distribution")
    axes[1, 0].set_xlabel("FICO Score")
    
    # 5. DTI Ratio
    sns.histplot(ax=axes[1, 1], data=df, x="dti", kde=True, color="#1abc9c", bins=30)
    axes[1, 1].set_title("Debt-to-Income (DTI) Distribution")
    axes[1, 1].set_xlabel("DTI Ratio")
    
    # 6. Home Ownership
    sns.countplot(ax=axes[1, 2], data=df, x="home_ownership", hue="home_ownership", legend=False, palette="Set2")
    axes[1, 2].set_title("Home Ownership Categories")
    axes[1, 2].set_xlabel("Home Ownership")
    axes[1, 2].tick_params(axis="x", rotation=30)
    
    save_figure(fig, "eda_univariate.png")
    print("Univariate analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_univariate_analysis()
