"""
Classification Model 01: Logistic Regression
Trains and evaluates a balanced L2-regularized Logistic Regression classifier.
"""

import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def train_logistic_regression():
    print("=" * 70)
    print(" CREDITFAIR CLASSIFICATION - MODEL 01: LOGISTIC REGRESSION")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, _, _ = get_prepared_data()
    
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    
    print("[INFO] Training Logistic Regression...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": "Logistic Regression",
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
    train_logistic_regression()
