# Diabetes Prediction with Machine Learning



## Project Overview

This project aims to predict diabetes occurrence using structured health data from the [BRFSS 2015 survey](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). It compares multiple machine learning models in terms of predictive power and interpretability.

The goal is to assist in early diagnosis and potential prevention of Type 2 diabetes, enabling data-driven support for public health interventions.



## Dataset

- Source: BRFSS 2015 (Kaggle)
- Binary target: `Diabetes_binary`
- Features: Health behaviours, conditions, demographics
- Sample size:  About  50,000  (balanced positive/negative)



## Models Used

- Logistic Regression (baseline + L1 selection)
- Random Forest (with GridSearchCV)
- XGBoost (with hyperparameter tuning)
- Stacking Classifier
- DummyClassifier 



## Feature Engineering

- Numerical features: scaled via `StandardScaler`
- Ordinal features: normalised
- L1-based feature selection using Logistic Regression
- SHAP values for model interpretability



## Evaluation Metrics

- Accuracy
- Cross-validation score (mean ± std)
- Confusion Matrix
- ROC AUC score and curve
- SHAP explanation plots

| Model               | Accuracy | CV Mean ± Std   |
| ------------------- | -------- | --------------- |
| Logistic Regression | 74.58%   | 74.78% ± 0.005  |
| Random Forest       | 74.82%   | 75.04% ± 0.0048 |
| XGBoost             | 75.31%   | 75.17% ± 0.0051 |
| Stacking Classifier | 74.98%   | –               |
| Dummy Classifier    | 50.06%   | 50.26% ± 0.00   |



## Visual Outputs

- Feature importance per model
- Confusion matrices
- ROC curve comparison
- SHAP summary and BMI dependence plots



## Project Files

| File               | Description                                  |
| ------------------ | -------------------------------------------- |
| `Data.py`          | Full training, evaluation, and SHAP pipeline |
| `requirements.txt` | Environment dependencies                     |



## How to Run

```bash
pip install -r requirements.txt  # optional if added
python Data.py
```

