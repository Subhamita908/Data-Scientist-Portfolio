

# Behavioral Credit Risk Prediction Using Temporal Payment Features

## 1. Objective

This project investigates whether short-term borrower default risk can be predicted from **behavioral payment dynamics** rather than static demographic or financial attributes. The research question is:

> *Do engineered temporal behavior features significantly improve 6-month default prediction accuracy compared to static baseline models?*

## 2. Dataset

Dataset: **Home Credit Default Risk** (public, Kaggle)

Contains month-level installment records, bill payments, credit utilization, and delinquency indicators for consumer loans.

## 3. Target Definition

Binary target:

* `Default = 1` if ≥90 days past due within the next **6 months**
* `Default = 0` otherwise

## 4. Research Hypothesis

Hypothesis: **Borrower behavior trends, volatility, and utilization patterns contain predictive signal for imminent default**, outperforming static models.
## 4.1 Folder Structure



Banking-Risk-Analytics/
    data/
    notebooks/
        01_EDA.ipynb
        02_Features.ipynb
        03_Models.ipynb
        04_Explainability.ipynb
        05_RiskDecisioning.ipynb
    src/
    dashboard/
        app.py (Streamlit)
    reports/
        academic_paper.pdf
        findings.pdf
        portfolio_summary.pdf
    README.md


## 5. Methodology

### 5.1 Preprocessing

* Missing value treatment
* Extreme value winsorization
* Normalization of financial magnitudes
* Temporal ordering by borrower ID

### 5.2 Temporal Feature Engineering

Rolling windows selected:

* **3-month**
* **6-month**
* **12-month**

For each window, compute:

* Delinquency frequency
* Delinquency days
* Payment-to-bill ratios
* Credit utilization (mean, max, slope)
* Balance momentum
* Volatility metrics (std, variance)
* Spending peaks
* Trend slopes via linear regression

### 5.3 Baseline Models

* Logistic Regression
* Random Forest

### 5.4 Behavior-Aware Models

* XGBoost
* LightGBM
* Optional: TabNet / TabTransformer

### 5.5 Evaluation Metrics

Reported metrics include:

* AUC-ROC
* Precision, Recall, F1
* Matthews Correlation Coefficient (MCC)
* KS Statistic
* Confusion Matrix

### 5.6 Validation Strategy

* Stratified train/test split by borrower
* Time-aware validation to preserve temporal leakage constraints

## 6. Results

Experimental results will compare:

1. **Static-only models**
2. **Temporal behavioral models**

Performance differences quantified across all metrics.

## 7. Contributions

* Original feature construction for behavioral credit modeling
* 6-month default prediction horizon
* Multi-window temporal aggregation
* Explainability via SHAP

## 8. Reproducibility

Implementation uses:

* Python
* Pandas
* Scikit-learn
* LightGBM/XGBoost

Notebooks and scripts included for replication.

## 9. Future Work

* Sequence modeling (LSTM/Transformer)
* Multi-horizon forecasting (3/6/12 months)
* Counterfactual explanations
* Portfolio-level expected loss simulation

## 10. References

Academic and industry literature to be cited.
