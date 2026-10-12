"""
Preprocessing Step 06: Feature Scaling & Pipeline Serialization
Fits StandardScaler and OneHotEncoder on training split to eliminate data leakage.
Persists the final production preprocessor artifact to models/preprocessor.pkl.
"""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import (
    load_dataset,
    engineer_credit_features,
    build_preprocessor,
    save_artifact,
    PREPROCESSOR_PATH,
    TARGET_COL,
)


def run_feature_scaling():
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 06: FEATURE SCALING & SERIALIZATION")
    print("=" * 70)
    
    df = engineer_credit_features(load_dataset())
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COL],
    )
    
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    preprocessor = build_preprocessor()
    print("[INFO] Fitting ColumnTransformer on Training Split...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print(f"X_train scaled matrix dimensions: {X_train_scaled.shape}")
    print(f"X_test scaled matrix dimensions : {X_test_scaled.shape}")
    
    # Persist artifact
    save_artifact(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor successfully serialized to: {PREPROCESSOR_PATH}")
    print("=" * 70)
    return preprocessor, X_train_scaled, X_test_scaled


if __name__ == "__main__":
    run_feature_scaling()
