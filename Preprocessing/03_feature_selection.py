"""
Preprocessing Step 03: Feature Selection
Ranks predictors using mutual information and tree importance to identify
the most discriminative features for credit default modeling.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, engineer_credit_features, NUMERIC_FEATURES, TARGET_COL


def run_feature_selection():
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 03: FEATURE SELECTION")
    print("=" * 70)
    
    raw_df = load_dataset()
    df = engineer_credit_features(raw_df)
    
    eval_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [
        "installment_to_income",
        "high_utilization_flag",
        "open_to_total_acc_ratio",
        "has_derogatory_record",
    ]
    
    X = df[eval_cols].fillna(df[eval_cols].median())
    y = df[TARGET_COL].values
    
    # 1. Mutual Information Ranking
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=eval_cols).sort_values(ascending=False)
    
    # 2. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    rf.fit(X, y)
    rf_series = pd.Series(rf.feature_importances_, index=eval_cols).sort_values(ascending=False)
    
    summary_df = pd.DataFrame({
        "Mutual Information": mi_series,
        "Tree Importance": rf_series,
    }).sort_values(by="Tree Importance", ascending=False)
    
    print("\n--- Feature Importance Ranking ---")
    print(summary_df.round(4))
    
    selected_features = list(summary_df.index[:12])
    print(f"\n[INFO] Top {len(selected_features)} Priority Risk Predictors:")
    for rank, feat in enumerate(selected_features, 1):
        print(f"  {rank:2d}. {feat}")
        
    print("Feature selection analysis successfully completed.")
    print("=" * 70)
    return summary_df


if __name__ == "__main__":
    run_feature_selection()
