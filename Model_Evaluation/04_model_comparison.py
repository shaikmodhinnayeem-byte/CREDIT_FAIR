"""
Model Evaluation Step 04: Model Comparison Benchmark
Evaluates Logistic Regression, Decision Tree, Random Forest, and XGBoost side-by-side.
Saves comparison plot to reports/figures/model_comparison.png.
"""

import sys
from pathlib import Path
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import save_figure


def run_model_comparison():
    print("=" * 70)
    print(" CREDITFAIR MODEL EVALUATION - STEP 04: MODEL COMPARISON BENCHMARK")
    print("=" * 70)
    
    models_to_run = [
        ("Logistic Regression", "Classification.01_logistic_regression", "train_logistic_regression"),
        ("Decision Tree", "Classification.02_decision_tree", "train_decision_tree"),
        ("Random Forest", "Classification.03_random_forest", "train_random_forest"),
        ("XGBoost", "Classification.04_xgboost", "train_xgboost"),
    ]
    
    results = []
    
    for name, module_path, func_name in models_to_run:
        print(f"\n[BENCHMARK] Executing {name}...")
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        _, metrics = func()
        results.append(metrics)
        
    df_comparison = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print(" CREDITFAIR MODEL COMPARISON LEADERBOARD")
    print("=" * 70)
    print(df_comparison.to_string(index=False))
    print("=" * 70)
    
    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CreditFair Classifier Benchmark Comparison", fontsize=15, weight="bold")
    
    # 1. ROC-AUC Comparison
    sns.barplot(
        data=df_comparison,
        x="Model",
        y="ROC-AUC",
        hue="Model",
        legend=False,
        palette="Blues_r",
        ax=axes[0],
    )
    axes[0].set_title("ROC-AUC Discrimination Power")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].tick_params(axis="x", rotation=15)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.04),
                         ha='center', va='center', color='white', weight='bold')
    
    # 2. F1 Score Comparison
    sns.barplot(
        data=df_comparison,
        x="Model",
        y="F1 Score",
        hue="Model",
        legend=False,
        palette="Greens_r",
        ax=axes[1],
    )
    axes[1].set_title("F1 Score (Balanced Default Detection)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=15)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.04),
                         ha='center', va='center', color='white', weight='bold')
    
    save_figure(fig, "model_comparison.png")
    print("Model comparison complete.")
    print("=" * 70)
    return df_comparison


if __name__ == "__main__":
    run_model_comparison()
