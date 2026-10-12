"""
CreditFair Shared Project Utilities and Configuration.
Provides paths, data loading, preprocessing pipelines, and figure persistence
used consistently across EDA, Preprocessing, Modeling, Fairness, and Deployment.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define Root and Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "creditfair_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature subsets definition
NUMERIC_FEATURES: List[str] = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_score",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "emp_length_years",
]

CATEGORICAL_FEATURES: List[str] = [
    "home_ownership",
    "verification_status",
    "purpose",
]

PROTECTED_ATTRIBUTES: List[str] = [
    "home_ownership",
    "income_bracket",
]

TARGET_COL: str = "loan_status"


def load_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads and standardizes the base CreditFair dataset."""
    path = csv_path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    
    # Standardize target
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].apply(
            lambda x: 1 if str(x).strip().lower() in ["charged off", "default", "1"] else 0
        ).astype(int)
            
    # Clean numeric fields if strings present
    if "int_rate" in df.columns:
        def parse_pct(x):
            if pd.isna(x): return np.nan
            s = str(x).replace("%", "").strip()
            try: return float(s)
            except: return np.nan
        df["int_rate"] = df["int_rate"].apply(parse_pct)
        
    if "revol_util" in df.columns:
        def parse_util(x):
            if pd.isna(x): return np.nan
            s = str(x).replace("%", "").strip()
            try: return float(s)
            except: return np.nan
        df["revol_util"] = df["revol_util"].apply(parse_util)
        
    if "emp_length" in df.columns and "emp_length_years" not in df.columns:
        def parse_emp(x):
            if pd.isna(x): return np.nan
            s = str(x).lower()
            if "< 1" in s: return 0.0
            if "10+" in s: return 10.0
            digits = "".join([c for c in s if c.isdigit()])
            return float(digits) if digits else np.nan
        df["emp_length_years"] = df["emp_length"].apply(parse_emp)
        
    if "term" in df.columns and "term_months" not in df.columns:
        df["term_months"] = df["term"].astype(str).apply(lambda x: 60 if "60" in x else 36)

    if "income_bracket" not in df.columns and "annual_inc" in df.columns:
        df["income_bracket"] = pd.qcut(
            df["annual_inc"].fillna(df["annual_inc"].median()),
            q=4,
            labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
        ).astype(str)

    return df


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates financial risk indicator features."""
    df_out = df.copy()
    
    # 1. Installment to Income Ratio
    annual_inc_safe = df_out["annual_inc"].replace(0, np.nan).fillna(df_out["annual_inc"].median())
    monthly_income = annual_inc_safe / 12.0
    df_out["installment_to_income"] = (df_out["installment"] / monthly_income).clip(0, 1.5)
    
    # 2. Credit line utilization risk flag
    revol_util_safe = df_out["revol_util"].fillna(df_out["revol_util"].median())
    df_out["high_utilization_flag"] = (revol_util_safe > 80.0).astype(int)
    
    # 3. Credit line depth
    total_acc_safe = df_out["total_acc"].replace(0, np.nan).fillna(df_out["total_acc"].median())
    df_out["open_to_total_acc_ratio"] = (df_out["open_acc"] / total_acc_safe).clip(0, 1.0)
    
    # 4. Derogatory event flag
    pub_rec = df_out["pub_rec"].fillna(0)
    bankruptcies = df_out["pub_rec_bankruptcies"].fillna(0)
    df_out["has_derogatory_record"] = ((pub_rec > 0) | (bankruptcies > 0)).astype(int)

    return df_out


def build_preprocessor() -> ColumnTransformer:
    """Builds an enterprise sklearn ColumnTransformer for numeric and categorical features."""
    num_cols = NUMERIC_FEATURES + [
        "installment_to_income",
        "high_utilization_flag",
        "open_to_total_acc_ratio",
        "has_derogatory_record",
    ]
    cat_cols = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_prepared_data(test_size: float = 0.20, random_state: int = 42):
    """
    Loads data, engineers features, performs stratified split, and fits preprocessor.
    Returns: X_train_proc, X_test_proc, y_train, y_test, preprocessor, train_df, test_df
    """
    raw_df = load_dataset()
    df = engineer_credit_features(raw_df)

    # Stratified split on target
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X_train_raw = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, train_df, test_df


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Saves a matplotlib figure cleanly to reports/figures/."""
    out_file = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE SAVED] -> {out_file}")
    return out_file


def save_artifact(obj: Any, file_path: Path) -> None:
    """Saves serialized artifact safely using pickle."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[ARTIFACT SAVED] -> {file_path}")


def load_artifact(file_path: Path) -> Any:
    """Loads serialized pickle artifact."""
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found at {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)
