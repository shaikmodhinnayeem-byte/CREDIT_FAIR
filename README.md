# CreditFair: End-to-End Fair Lending & Credit Default Risk Architecture

A production-grade machine learning and regulatory algorithmic fairness framework designed to predict credit default risk while auditing and mitigating disparate demographic impacts across protected and socio-economic subgroups.

---

## 📁 Project Directory Structure

```
CreditFair/
│
├── dataset/
│   └── data.csv
│
├── EDA/
│   ├── 01_data_understanding.py
│   ├── 02_univariate_analysis.py
│   ├── 03_bivariate_analysis.py
│   └── 04_multivariate_analysis.py
│
├── Preprocessing/
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py
│   ├── 03_feature_selection.py
│   ├── 04_encoding.py
│   ├── 05_train_test_split.py
│   └── 06_feature_scaling.py
│
├── Classification/
│   ├── 01_logistic_regression.py
│   ├── 02_decision_tree.py
│   ├── 03_random_forest.py
│   └── 04_xgboost.py
│
├── Model_Evaluation/
│   ├── 01_model_metrics.py
│   ├── 02_confusion_matrix.py
│   ├── 03_roc_curve.py
│   └── 04_model_comparison.py
│
├── Model_Improvement/
│   ├── 01_hyperparameter_tuning.py
│   ├── 02_cross_validation.py
│   └── 03_improved_model.py
│
├── Fairness/
│   ├── 01_group_analysis.py
│   ├── 02_fairness_metrics.py
│   └── 03_fairness_visualization.py
│
├── Deployment/
│   ├── 01_load_model.py
│   └── 02_prediction.py
│
├── models/
│   ├── creditfair_model.pkl
│   └── preprocessor.pkl
│
├── reports/
│   └── figures/
│
├── main.py
├── project_utils.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Execution

### 1. Installation
Install core dependencies using Python 3.10+:
```bash
pip install -r requirements.txt
```

### 2. End-to-End Master Pipeline
Execute all stages sequentially (Data Ingestion -> EDA -> Preprocessing -> Classification -> Evaluation -> Model Improvement & Calibration -> Fairness Audit -> Deployment Verification):
```bash
python main.py
```

### 3. Launch Web Application (No Streamlit)
Launch the native web application with zero third-party framework dependencies:
```bash
python server.py
```
Open **`http://localhost:8000`** in your browser to interact with all 9 dashboard screens.

---

## 🔬 Modular Pipeline Breakdown

Each phase is completely modular and can be run independently:

### 📊 Exploratory Data Analysis (EDA)
- **`EDA/01_data_understanding.py`**: Shape auditing, type checking, missing value ratios, and target imbalance analysis.
- **`EDA/02_univariate_analysis.py`**: Distribution plots for FICO scores, debt-to-income, loan amounts, and categories (`reports/figures/eda_univariate.png`).
- **`EDA/03_bivariate_analysis.py`**: Default rates across borrower demographics and income tiers (`reports/figures/eda_bivariate.png`).
- **`EDA/04_multivariate_analysis.py`**: Correlation heatmaps across financial predictors (`reports/figures/eda_correlation_matrix.png`).

### ⚙️ Preprocessing & Feature Engineering
- **`Preprocessing/01_data_cleaning.py`**: Median imputation for numerical fields, mode imputation for categorical attributes, percentage normalization.
- **`Preprocessing/02_feature_engineering.py`**: Financial domain indicators (`installment_to_income`, `high_utilization_flag`, `open_to_total_acc_ratio`, `has_derogatory_record`).
- **`Preprocessing/03_feature_selection.py`**: Mutual information and tree importance ranking for risk feature selection.
- **`Preprocessing/04_encoding.py`**: One-Hot Encoding for multi-class categorical features.
- **`Preprocessing/05_train_test_split.py`**: Stratified 80/20 train-test partitioning ensuring target and demographic alignment.
- **`Preprocessing/06_feature_scaling.py`**: Leak-free `StandardScaler` pipeline fitting and artifact persistence to `models/preprocessor.pkl`.

### 🤖 Classification Models
- **`Classification/01_logistic_regression.py`**: Balanced L2-regularized logistic regression benchmark.
- **`Classification/02_decision_tree.py`**: Cost-sensitive decision tree classifier with pruning.
- **`Classification/03_random_forest.py`**: Balanced Random Forest ensemble classifier.
- **`Classification/04_xgboost.py`**: Extreme Gradient Boosting with class imbalance scaling (`scale_pos_weight`).

### 📈 Model Evaluation
- **`Model_Evaluation/01_model_metrics.py`**: Computes Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Brier Score.
- **`Model_Evaluation/02_confusion_matrix.py`**: Generates normalized and count-based confusion matrices (`reports/figures/confusion_matrix.png`).
- **`Model_Evaluation/03_roc_curve.py`**: Generates Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves (`reports/figures/roc_curve.png`).
- **`Model_Evaluation/04_model_comparison.py`**: Benchmarks all 4 classifiers side-by-side and plots comparative leaderboards (`reports/figures/model_comparison.png`).

### 🔧 Model Improvement & Calibration
- **`Model_Improvement/01_hyperparameter_tuning.py`**: Randomized search over learning rates, tree depth, and subsample ratios.
- **`Model_Improvement/02_cross_validation.py`**: 5-Fold Stratified Cross-Validation checking performance variance.
- **`Model_Improvement/03_improved_model.py`**: Trains final champion model with Platt Scaling probability calibration and saves to `models/creditfair_model.pkl`.

### ⚖️ Algorithmic Fairness & Regulatory Auditing
- **`Fairness/01_group_analysis.py`**: Evaluates approval and default rates across protected attributes (`home_ownership`, `income_bracket`).
- **`Fairness/02_fairness_metrics.py`**: Audits Disparate Impact Ratio (DIR, Four-Fifths 80% Rule), Statistical Parity Difference (SPD), and Equal Opportunity Difference (EOD).
- **`Fairness/03_fairness_visualization.py`**: Generates fairness audit charts saved to `reports/figures/fairness_visualization.png`.

### 🚀 Production Deployment
- **`Deployment/01_load_model.py`**: Verifies and loads `models/preprocessor.pkl` and `models/creditfair_model.pkl`.
- **`Deployment/02_prediction.py`**: Real-time scoring function `predict_applicant(applicant_dict)` returning loan decisions, default probability, and risk tiers (`Prime`, `Near-Prime`, `Subprime`).

---

## 📊 Summary of Generated Artifacts & Reports

| Artifact Path | Description |
|---|---|
| `dataset/data.csv` | Cleaned base credit lending dataset |
| `models/preprocessor.pkl` | Serialized scikit-learn feature preprocessor |
| `models/creditfair_model.pkl` | Serialized calibrated production credit scoring model |
| `reports/figures/eda_univariate.png` | Univariate distributions of credit metrics |
| `reports/figures/eda_bivariate.png` | Bivariate analysis across demographic groups |
| `reports/figures/eda_correlation_matrix.png` | Feature correlation matrix heatmap |
| `reports/figures/confusion_matrix.png` | Confusion matrix audit |
| `reports/figures/roc_curve.png` | ROC and Precision-Recall curves |
| `reports/figures/model_comparison.png` | Classifier benchmark leaderboard |
| `reports/figures/fairness_visualization.png` | Fair lending subgroup audit visualization |
