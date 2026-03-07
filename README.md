# MSIS-522-HW1

## Orange Juice Demand Prediction

This project implements a full data science workflow for predicting
orange juice sales.

The dataset contains weekly store-level observations including
price, promotions, brand information, and demographic variables.

The target variable is **logmove**, which represents the log of
units sold.

### Main Steps

1. Descriptive analysis to understand the dataset
2. Training several machine learning models
3. Comparing model performance
4. Using SHAP to explain predictions
5. Deploying the results in a Streamlit application

### Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- LightGBM
- Neural Network (MLP)

### Run the Project

Train models:
python src/train_models.py

Run the Streamlit app:
streamlit run app.py

## Streamlit App

The Streamlit app contains four sections:
1. Executive Summary
2. Descriptive Analytics
3. Model Performance
4. Explainability & Interactive Prediction
