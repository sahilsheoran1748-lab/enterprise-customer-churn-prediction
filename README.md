# Enterprise Customer Churn Prediction

## Project Overview

This project develops a supervised machine learning system for predicting customer churn in the telecommunications domain.

The objective is to identify customers who are more likely to discontinue their services and provide a structured risk classification that can support customer retention analysis.

The project includes data cleaning, feature engineering, model development, model comparison, hyperparameter optimization using Optuna, explainable AI using SHAP, batch prediction, and a reusable machine learning pipeline.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Dataset characteristics:

- 7,043 customer records
- 21 original columns
- Target variable: Churn
- Target classes: Yes and No

## Data Preparation

The following preprocessing steps were performed:

1. Dataset quality inspection
2. Missing-value analysis
3. Duplicate-row verification
4. Conversion of TotalCharges to numeric format
5. Median imputation for missing TotalCharges
6. Removal of customerID
7. Feature engineering
8. Numerical feature standardization
9. Categorical feature one-hot encoding

## Engineered Features

- AvgMonthlySpend
- TotalServicesActive

## Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Tuned XGBoost

Optuna was used for XGBoost hyperparameter optimization with 50 trials and 5-fold stratified cross-validation.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8105 | 0.6801 | 0.5401 | 0.6021 | 0.8472 |
| Random Forest | 0.7793 | 0.6090 | 0.4706 | 0.5309 | 0.8202 |
| XGBoost | 0.7999 | 0.6544 | 0.5214 | 0.5804 | 0.8431 |
| Tuned XGBoost | 0.8062 | 0.6797 | 0.5107 | 0.5832 | 0.8488 |

The tuned XGBoost model achieved the highest ROC-AUC among the evaluated models.

Logistic Regression achieved slightly higher accuracy, precision, recall, and F1 score on this particular test split.

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to analyze model feature contributions and improve interpretability.

The project includes:

- SHAP summary visualization
- SHAP feature importance data
- Feature importance visualization
- Feature importance CSV output

## Prediction System

The saved machine learning pipeline can generate:

- Churn probability
- Churn prediction
- Customer risk level
- Recommended action category

Risk classification:

- High Risk: probability >= 70%
- Medium Risk: probability >= 50% and < 70%
- Low Risk: probability < 50%

These thresholds are configurable and should be validated against business objectives before production use.

## Project Outputs

The project generates:

- ROC-AUC comparison chart
- Tuned XGBoost confusion matrix
- Feature importance chart
- Feature importance CSV
- SHAP summary chart
- SHAP importance CSV
- Batch prediction CSV
- Model comparison CSV
- Final model evaluation CSV

## Project Structure

churn_project/
|
|-- data/
|-- models/
|   |-- churn_prediction_pipeline.joblib
|   |-- metadata.json
|-- notebooks/
|-- outputs/
|-- reports/
|-- src/
|-- tests/
|-- requirements.txt
|-- README.md

## Installation

Install the required dependencies:

pip install -r requirements.txt

## Model Usage

The trained pipeline is stored at:

models/churn_prediction_pipeline.joblib

The pipeline contains preprocessing and the trained model so that new customer records can be processed consistently.

## Project Objective

This project demonstrates an end-to-end machine learning workflow covering:

- Data preprocessing
- Feature engineering
- Classification
- Model evaluation
- Hyperparameter optimization
- Explainable AI
- Model persistence
- Customer-level prediction
- Batch prediction
- Reproducible project organization

## Important Note

The reported metrics are based on the current train/test evaluation and should not be interpreted as guaranteed production performance.

Before real-world deployment, additional validation, threshold optimization, monitoring, fairness checks, data-drift analysis, and business validation would be required.

## Author

Sahil Kumar

AI & Machine Learning Project

Purpose: Internship and Professional Portfolio
