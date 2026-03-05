import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.preprocessing import load_data

st.set_page_config(layout="wide")

df = load_data()

tabs = st.tabs([
"Executive Summary",
"Descriptive Analytics",
"Model Performance",
"Explainability & Prediction"
])


with tabs[0]:

    st.title("Orange Juice Sales Forecasting")

    st.write("""
    In this project, I try to predict orange juice sales using machine learning.

    The dataset includes information about price, promotion,
    brand characteristics, and some demographic variables.

    The target variable is **logmove**, which represents the log of
    units sold in a given week.

    My goal is to understand which factors influence sales
    and which model gives the most accurate prediction.
    """)


with tabs[1]:

    st.header("Descriptive Analytics")

    fig, ax = plt.subplots()
    sns.histplot(df["logmove"], kde=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="brand", y="logmove", data=df, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(x="price", y="logmove", data=df, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


with tabs[2]:

    st.header("Model Performance")

    results = pd.read_csv("model_results.csv")

    st.dataframe(results)

    fig, ax = plt.subplots()

    sns.barplot(data=results, x="model", y="RMSE", ax=ax)

    st.pyplot(fig)


with tabs[3]:

    st.header("Interactive Prediction")

    model_name = st.selectbox(
        "Select Model",
        ["linear","ridge","lasso","cart","random_forest","lightgbm"]
    )

    model = joblib.load(f"models/{model_name}.pkl")

    price = st.slider("Price",0.5,5.0,2.0)
    promotion = st.selectbox("Promotion",[0,1])

    input_df = pd.DataFrame({
        "price":[price],
        "feat":[promotion]
    })

    prediction = model.predict(input_df)[0]

    st.write("Predicted log sales:",prediction)
