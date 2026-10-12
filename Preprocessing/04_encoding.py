"""
Preprocessing Step 04: Categorical Encoding
Applies One-Hot Encoding to categorical loan attributes (home_ownership,
verification_status, purpose) with unknown handling.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_dataset, CATEGORICAL_FEATURES


def run_encoding():
    print("=" * 70)
    print(" CREDITFAIR PREPROCESSING - STEP 04: CATEGORICAL ENCODING")
    print("=" * 70)
    
    df = load_dataset()
    cat_df = df[CATEGORICAL_FEATURES].fillna("Missing")
    
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_matrix = encoder.fit_transform(cat_df)
    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    
    encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_columns)
    print(f"Original Categoricals: {CATEGORICAL_FEATURES}")
    print(f"Encoded Features Count: {encoded_df.shape[1]}")
    print("\n--- Encoded Feature Columns ---")
    for col in encoded_columns:
        print(f"  * {col}")
        
    print("\n--- Sample Encoded Head ---")
    print(encoded_df.head(3))
    
    print("Categorical encoding demonstration completed.")
    print("=" * 70)
    return encoder, encoded_df


if __name__ == "__main__":
    run_encoding()
