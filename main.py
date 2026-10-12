"""
CreditFair: End-to-End Orchestrator Pipeline
Executes complete automated machine learning, fairness auditing, and deployment workflow:
  1. Data Ingestion & Exploratory Data Analysis (EDA)
  2. Preprocessing, Feature Engineering & Scaling
  3. Classification Modeling (Logistic Regression, Decision Tree, Random Forest, XGBoost)
  4. Model Evaluation & Benchmark Comparison
  5. Hyperparameter Improvement & Probability Calibration
  6. Algorithmic Fairness Audit & Compliance Verification
  7. Artifact Persistence & Deployment Testing
"""

import sys
import os
from pathlib import Path
import importlib
import time

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("      CREDITFAIR: FAIR & ACCURATE MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Working Directory: {PROJECT_ROOT}\n")

    # -------------------------------------------------------------
    # STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 80)
    
    eda_01 = importlib.import_module("EDA.01_data_understanding")
    eda_01.run_data_understanding()

    eda_02 = importlib.import_module("EDA.02_univariate_analysis")
    eda_02.run_univariate_analysis()

    eda_03 = importlib.import_module("EDA.03_bivariate_analysis")
    eda_03.run_bivariate_analysis()

    eda_04 = importlib.import_module("EDA.04_multivariate_analysis")
    eda_04.run_multivariate_analysis()

    # -------------------------------------------------------------
    # STAGE 2: PREPROCESSING & FEATURE ENGINEERING
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 2: PREPROCESSING & FEATURE ENGINEERING")
    print("#" * 80)
    
    prep_01 = importlib.import_module("Preprocessing.01_data_cleaning")
    prep_01.run_data_cleaning()

    prep_02 = importlib.import_module("Preprocessing.02_feature_engineering")
    prep_02.run_feature_engineering()

    prep_03 = importlib.import_module("Preprocessing.03_feature_selection")
    prep_03.run_feature_selection()

    prep_04 = importlib.import_module("Preprocessing.04_encoding")
    prep_04.run_encoding()

    prep_05 = importlib.import_module("Preprocessing.05_train_test_split")
    prep_05.run_train_test_split()

    prep_06 = importlib.import_module("Preprocessing.06_feature_scaling")
    prep_06.run_feature_scaling()

    # -------------------------------------------------------------
    # STAGE 3: CLASSIFICATION BENCHMARKS
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 3: CLASSIFICATION BENCHMARKS")
    print("#" * 80)

    clf_01 = importlib.import_module("Classification.01_logistic_regression")
    clf_01.train_logistic_regression()

    clf_02 = importlib.import_module("Classification.02_decision_tree")
    clf_02.train_decision_tree()

    clf_03 = importlib.import_module("Classification.03_random_forest")
    clf_03.train_random_forest()

    clf_04 = importlib.import_module("Classification.04_xgboost")
    clf_04.train_xgboost()

    # -------------------------------------------------------------
    # STAGE 4: MODEL EVALUATION & COMPARISON
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 4: MODEL EVALUATION & COMPARISON")
    print("#" * 80)

    eval_04 = importlib.import_module("Model_Evaluation.04_model_comparison")
    eval_04.run_model_comparison()

    eval_02 = importlib.import_module("Model_Evaluation.02_confusion_matrix")
    eval_03 = importlib.import_module("Model_Evaluation.03_roc_curve")
    from project_utils import get_prepared_data
    xgb_model, _ = clf_04.train_xgboost()
    _, X_test, _, y_test, _, _, _ = get_prepared_data()
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    eval_02.plot_confusion_matrix(y_test, y_pred)
    eval_03.plot_roc_pr_curves(y_test, y_prob)

    # -------------------------------------------------------------
    # STAGE 5: MODEL IMPROVEMENT & CALIBRATION
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 5: MODEL IMPROVEMENT & CALIBRATION")
    print("#" * 80)

    imp_02 = importlib.import_module("Model_Improvement.02_cross_validation")
    imp_02.run_cross_validation()

    imp_03 = importlib.import_module("Model_Improvement.03_improved_model")
    imp_03.train_improved_model()

    # -------------------------------------------------------------
    # STAGE 6: FAIRNESS AUDITING & COMPLIANCE
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 6: FAIRNESS AUDITING & COMPLIANCE")
    print("#" * 80)

    fair_01 = importlib.import_module("Fairness.01_group_analysis")
    fair_01.run_group_analysis()

    fair_02 = importlib.import_module("Fairness.02_fairness_metrics")
    fair_02.run_fairness_metrics()

    fair_03 = importlib.import_module("Fairness.03_fairness_visualization")
    fair_03.plot_fairness_audit()

    # -------------------------------------------------------------
    # STAGE 7: PRODUCTION DEPLOYMENT & INFERENCE TESTING
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> STAGE 7: DEPLOYMENT VERIFICATION & SAMPLE INFERENCE")
    print("#" * 80)

    dep_01 = importlib.import_module("Deployment.01_load_model")
    dep_01.run_load_verification()

    dep_02 = importlib.import_module("Deployment.02_prediction")
    dep_02.run_sample_predictions()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" [SUCCESS] CREDITFAIR END-TO-END PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    print(" All models, preprocessors, and figures generated.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
