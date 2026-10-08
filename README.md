# Telco Customer Churn Prediction

End-to-end machine learning project that predicts whether a telecom customer will churn, using the public [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset.

## Overview

The notebook covers a full classification pipeline: data cleaning, exploratory analysis, feature engineering, class-imbalance handling, model training, and evaluation. Results are interpreted with SHAP to highlight the main drivers of churn (for example contract type, tenure, and charges).

## Dataset

- **Source:** Kaggle — Blastchar Telco Customer Churn  
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Size:** 7,043 customers, 21 columns  
- **Target:** `Churn` (Yes / No)

Features include demographics, subscribed services, contract and billing details, tenure, and monthly/total charges.

## Approach

1. **Data cleaning** — Convert `TotalCharges` to numeric, handle missing values, drop `customerID`, encode the target.  
2. **EDA** — Target imbalance, correlation heatmap, and churn rates by contract and payment method.  
3. **Feature engineering** — Charge ratio and average monthly charge; one-hot encoding for categoricals; `StandardScaler` for numerics.  
4. **Split & imbalance** — Stratified 80/20 train–test split; SMOTE on training data; `class_weight` / `scale_pos_weight` for tree models.  
5. **Modeling** — Logistic Regression (baseline), Random Forest, and XGBoost.  
6. **Evaluation** — Confusion matrix, classification report (precision/recall), ROC-AUC, and SHAP feature importance.

On the held-out test set, XGBoost reached **ROC-AUC ≈ 0.83**, slightly above Logistic Regression.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

Open `Telco_Customer_Churn_Prediction.ipynb` in Jupyter and run all cells. Keep the CSV in the same folder as the notebook.

## Tech stack

Python, Jupyter, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, XGBoost, SHAP
