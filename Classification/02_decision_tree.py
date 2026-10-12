"""
Classification Model 02: Decision Tree
Trains and evaluates a cost-sensitive Decision Tree classifier.
"""

import sys
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import get_prepared_data


def train_decision_tree():
    print("=" * 70)
    print(" CREDITFAIR CLASSIFICATION - MODEL 02: DECISION TREE")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, _, _ = get_prepared_data()
    
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    
    print("[INFO] Training Decision Tree Classifier...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": "Decision Tree",
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
    train_decision_tree()
