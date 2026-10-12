"""
Classification Model 03: Random Forest
Trains and evaluates an ensemble Random Forest classifier with balanced class weighting.
"""

import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def train_random_forest():
    print("=" * 70)
    print(" CREDITFAIR CLASSIFICATION - MODEL 03: RANDOM FOREST")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, _, _ = get_prepared_data()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    
    print("[INFO] Training Random Forest Ensemble...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    
    print("\n--- Test Partition Performance ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  * {k:<15}: {v:.4f}")
            
    print("=" * 70)
    return model, metrics


if __name__ == "__main__":
    train_random_forest()
