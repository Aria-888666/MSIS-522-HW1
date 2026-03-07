import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import load_data

df = pd.read_csv("oj.csv")
feature_columns = df.drop(columns=["logmove"]).columns
input_df = df.drop(columns=["logmove"]).select_dtypes(include=np.number).mean().to_frame().T
tabs = st.tabs([
"Executive Summary",
"Descriptive Analytics",
"Model Performance",
"Explainability & Prediction"
])

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(page_title="Orange Juice Demand Prediction", layout="wide")

# Load dataset
df = pd.read_csv("oj.csv")

tabs = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction"
])

# -----------------------------
# TAB 1 — EXECUTIVE SUMMARY
# -----------------------------

with tabs[0]:

    st.title("Orange Juice Demand Prediction")

    st.markdown("""
### Business Problem

The goal of this project is to predict orange juice sales using machine learning techniques.

Using a retail dataset containing product pricing, promotional information, and demographic data, several predictive models were trained and evaluated.

The results show that tree-based models such as Random Forest and LightGBM perform better than simple linear models. These models capture nonlinear relationships between price, promotions, and consumer demand.

Explainability analysis using SHAP reveals that price and promotional activity are the most important drivers of sales.

Overall, this project demonstrates how machine learning can be used to support retail demand forecasting and provide insights into consumer purchasing behavior.
""")

# -----------------------------
# TAB 2 — DESCRIPTIVE ANALYTICS
# -----------------------------

with tabs[1]:

    st.header("Dataset Overview")

    st.markdown("""
This dataset contains weekly orange juice sales information across multiple stores.
The target variable **logmove** represents the logarithm of units sold.

Key features include:

- price
- promotional features
- brand identity
- store characteristics
- demographic variables
""")

    st.subheader("Target Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["logmove"], kde=True, ax=ax)
    ax.set_title("Distribution of logmove")
    st.pyplot(fig)

    st.markdown("""
The distribution of logmove appears relatively smooth and slightly right-skewed.
Most observations correspond to moderate sales volumes.
""")

    st.subheader("Brand vs Sales")

    fig, ax = plt.subplots()
    sns.boxplot(x="brand", y="logmove", data=df, ax=ax)
    st.pyplot(fig)

    st.markdown("Different brands show different sales distributions.")

    st.subheader("Price vs Sales")

    fig, ax = plt.subplots()
    sns.scatterplot(x="price", y="logmove", data=df, ax=ax)
    st.pyplot(fig)

    st.markdown("Higher prices generally correspond to lower sales.")

    st.subheader("Promotion vs Sales")

    fig, ax = plt.subplots()
    sns.boxplot(x="feat", y="logmove", data=df, ax=ax)
    st.pyplot(fig)

    st.markdown("Promotional features significantly increase sales.")

    st.subheader("Feature Relationships")

    fig = sns.pairplot(df[["price","logmove","AGE60","INCOME"]])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------
# TAB 3 — MODEL PERFORMANCE
# -----------------------------

with tabs[2]:

    st.header("Model Performance Comparison")

    if os.path.exists("model_results.csv"):
        results = pd.read_csv("model_results.csv")
        st.dataframe(results)

        st.subheader("RMSE Comparison")

        fig, ax = plt.subplots()
        sns.barplot(data=results, x="Model", y="RMSE", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.warning("model_results.csv not found. Please run train_models.py.")

    st.markdown("""
### Discussion

Tree-based models generally outperform linear models. Random Forest and LightGBM
typically achieve the lowest RMSE values.

This suggests that the relationship between features and sales is nonlinear.
""")

# -----------------------------
# TAB 4 — EXPLAINABILITY & PREDICTION
# -----------------------------

with tabs[3]:

    st.header("Interactive Prediction")

    model_name = st.selectbox(
        "Select Model",
        ["linear","ridge","lasso","cart","random_forest","lightgbm","mlp"]
    )

    model_path = f"models/{model_name}.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found.")
    else:

        model = joblib.load(model_path)

        # Create default input template
        X_template = df.drop(columns=["logmove"]).mean().to_frame().T

        price = st.slider("Price",0.5,5.0,2.0)
        feat = st.selectbox("Promotion",[0,1])

        if "price" in X_template.columns:
            X_template["price"] = price

        if "feat" in X_template.columns:
            X_template["feat"] = feat

        prediction = model.predict(X_template)[0]

        st.subheader("Predicted Log Sales")

        st.write(prediction)

    st.header("Model Explainability (SHAP)")

    if os.path.exists("models/random_forest.pkl"):

        model = joblib.load("models/random_forest.pkl")

        X = df.drop(columns=["logmove"])

        explainer = shap.Explainer(model)

        shap_values = explainer(X.sample(200))

        st.subheader("SHAP Summary Plot")

        fig = plt.figure()
        shap.summary_plot(shap_values, X.sample(200), show=False)
        st.pyplot(fig)

        st.subheader("SHAP Feature Importance")

        fig = plt.figure()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

    else:

        st.warning("Random Forest model required for SHAP analysis.")
