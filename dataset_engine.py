"""
CreditFair Dynamic Dataset Engine
Analyzes dataset/data.csv and extracts complete statistical, distribution,
outlier, and target-correlation profiles for EVERY feature in the dataset.
Outputs: assets/creditfair_dataset_features.json and dynamic API helpers.
"""

import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = ASSETS_DIR / "creditfair_dataset_features.json"


def clean_and_load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Strip % from int_rate and revol_util if present
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
            
    # Standardize string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df


def calculate_feature_profiles():
    df = clean_and_load_data()
    total_rows, total_cols = df.shape
    
    # Target breakdown
    target_col = "loan_status"
    has_target = target_col in df.columns
    target_counts = df[target_col].value_counts().to_dict() if has_target else {}
    target_pcts = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict() if has_target else {}
    is_default = (df[target_col] == "Charged Off").astype(int) if has_target else None

    # Identify numeric and categorical columns
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    features_data = {}

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isnull().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        valid_count = int(s.notnull().sum())
        is_num = col in numeric_cols

        feat_info = {
            "name": col,
            "type": "Numeric" if is_num else "Categorical",
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "valid_count": valid_count,
            "cardinality": int(s.nunique(dropna=True)),
        }

        if is_num:
            valid_s = s.dropna()
            if len(valid_s) > 0:
                mean_val = float(valid_s.mean())
                std_val = float(valid_s.std())
                median_val = float(valid_s.median())
                min_val = float(valid_s.min())
                max_val = float(valid_s.max())
                q25 = float(valid_s.quantile(0.25))
                q75 = float(valid_s.quantile(0.75))
                iqr = q75 - q25
                lower_fence = q25 - 1.5 * iqr
                upper_fence = q75 + 1.5 * iqr
                
                outliers = valid_s[(valid_s < lower_fence) | (valid_s > upper_fence)]
                outlier_count = int(len(outliers))
                outlier_pct = round((outlier_count / len(valid_s)) * 100, 2)

                # Histogram bins (12 bins)
                counts, bin_edges = np.histogram(valid_s, bins=12)
                hist_bins = []
                for i in range(len(counts)):
                    hist_bins.append({
                        "bin_start": round(float(bin_edges[i]), 2),
                        "bin_end": round(float(bin_edges[i+1]), 2),
                        "count": int(counts[i]),
                        "pct": round(float(counts[i] / len(valid_s) * 100), 2)
                    })

                # Target breakdown for numeric
                target_breakdown = {}
                if has_target:
                    for t_val in ["Fully Paid", "Charged Off"]:
                        sub_s = df[df[target_col] == t_val][col].dropna()
                        if len(sub_s) > 0:
                            target_breakdown[t_val] = {
                                "count": int(len(sub_s)),
                                "mean": round(float(sub_s.mean()), 2),
                                "median": round(float(sub_s.median()), 2),
                                "std": round(float(sub_s.std()), 2)
                            }

                # Correlation with default
                corr_target = 0.0
                if has_target and is_default is not None:
                    corr_val = valid_s.corr(is_default.loc[valid_s.index])
                    corr_target = round(float(corr_val), 4) if not math.isnan(corr_val) else 0.0

                feat_info["stats"] = {
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "median": round(median_val, 2),
                    "min": round(min_val, 2),
                    "max": round(max_val, 2),
                    "q25": round(q25, 2),
                    "q75": round(q75, 2),
                    "iqr": round(iqr, 2),
                    "lower_fence": round(lower_fence, 2),
                    "upper_fence": round(upper_fence, 2),
                    "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                    "corr_with_default": corr_target
                }
                feat_info["histogram"] = hist_bins
                feat_info["target_breakdown"] = target_breakdown
        else:
            # Categorical stats
            valid_s = s.dropna()
            vc = valid_s.value_counts()
            top_cats = []
            target_breakdown = {}

            # Limit to top 15 categories for neat display
            for cat, cnt in vc.head(15).items():
                cat_str = str(cat)
                pct = round(float(cnt / len(valid_s) * 100), 2)
                top_cats.append({
                    "category": cat_str,
                    "count": int(cnt),
                    "pct": pct
                })
                if has_target:
                    cat_mask = (df[col] == cat)
                    sub_target = df.loc[cat_mask, target_col]
                    def_count = int((sub_target == "Charged Off").sum())
                    fp_count = int((sub_target == "Fully Paid").sum())
                    total_cat = def_count + fp_count
                    def_rate = round((def_count / total_cat * 100), 2) if total_cat > 0 else 0.0
                    target_breakdown[cat_str] = {
                        "fully_paid": fp_count,
                        "charged_off": def_count,
                        "default_rate_pct": def_rate
                    }

            feat_info["stats"] = {
                "top_value": str(vc.index[0]) if len(vc) > 0 else "N/A",
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round(float(vc.iloc[0] / len(valid_s) * 100), 2) if len(vc) > 0 else 0.0,
                "unique_categories": int(len(vc))
            }
            feat_info["categories"] = top_cats
            feat_info["target_breakdown"] = target_breakdown

        features_data[col] = feat_info

    # Overall summary
    dataset_summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "numerical_columns_count": len(numeric_cols),
        "categorical_columns_count": len(categorical_cols),
        "numerical_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target_col": target_col,
        "target_distribution": {
            "counts": target_counts,
            "percentages": target_pcts
        },
        "features": features_data
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    print(f"[SUCCESS] Processed {len(features_data)} features from CreditFair dataset.")
    print(f"[OUTPUT] Saved feature profiles to {OUTPUT_JSON}")
    return dataset_summary


if __name__ == "__main__":
    calculate_feature_profiles()
