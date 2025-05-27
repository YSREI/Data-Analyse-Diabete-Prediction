# Diabetes Prediction with Machine Learning

## Project Overview

This project aims to predict diabetes occurrence using structured health data from the [BRFSS 2015 survey](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). It compares multiple machine learning models in terms of predictive power and interpretability.

The goal is to assist in early diagnosis and potential prevention of Type 2 diabetes, enabling data-driven support for public health interventions.



## Dataset

- Source: BRFSS 2015 (Kaggle)
- Binary target: `Diabetes_binary`
- Features: Health behaviours, conditions, demographics
- Sample size:  About  50,000  (balanced positive/negative)

## Preprocessing

- Standardised numerical features using `StandardScaler`
- Normalised ordinal features (scaled to [0, 1])
- No missing values, clean categorical/binary inputs
- Train-validation-test split: 60%–20%–20% with stratification

## Models Implemented

| Model               | Test Accuracy | Cross-Validation |
| ------------------- | ------------- | ---------------- |
| Logistic Regression | 74.58%        | 74.78% ± 0.50%   |
| Random Forest       | 74.82%        | 75.04% ± 0.48%   |
| XGBoost             | 75.31%        | 75.17% ± 0.51%   |
| Stacking Classifier | 74.98%        | —                |
| Dummy Baseline      | 50.06%        | 50.26% ± 0.00%   |

All models were tuned using `GridSearchCV` and evaluated with accuracy, confusion matrices, and ROC curves.

## Technologies Used

- Python 3.10+
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn
- Pandas / NumPy

## Future Work

- Incorporate deep learning for temporal tracking of health indicators
- Use external datasets for cross-population validation
- Explore longitudinal risk modelling to assess progression from prediabetes to diabetes



