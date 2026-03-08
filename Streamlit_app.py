import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(page_title="Orange Juice Demand Prediction", layout="wide")

@st.cache_data
def load_df():
    """
    Load the OJ dataset. Tries local file first; falls back to downloading
    from a public URL if the local file is missing or empty (e.g. Git LFS issue).
    """
    local_path = "oj.csv"

    # Check local file is non-empty
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return pd.read_csv(local_path)

    # Fallback: download from public source
    import urllib.request
    url = "https://raw.githubusercontent.com/cran/bayesm/master/data-raw/orangeJuice.csv"
    fallback_path = "/tmp/oj.csv"
    try:
        urllib.request.urlretrieve(url, fallback_path)
        raw = pd.read_csv(fallback_path)
        # Standardise column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]
        # Rename to match expected schema if needed
        rename_map = {}
        if "move"    in raw.columns and "logmove" not in raw.columns:
            raw["logmove"] = np.log(raw["move"])
        if "store"   in raw.columns: rename_map["store"]   = "store"
        raw = raw.rename(columns=rename_map)
        return raw
    except Exception as e:
        st.error(
            f"Could not load oj.csv locally or from fallback URL.\n\n"
            f"Please make sure oj.csv is committed to your GitHub repo as a real file "
            f"(not a Git LFS pointer). Error: {e}"
        )
        st.stop()

df = load_df()

# Identify feature columns (everything except target)
FEATURE_COLS = [c for c in df.columns if c != "logmove"]

# Identify numeric vs categorical columns
NUM_COLS = df[FEATURE_COLS].select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df[FEATURE_COLS].select_dtypes(exclude=[np.number]).columns.tolist()

tabs = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ─────────────────────────────────────────
# TAB 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────

with tabs[0]:

    st.title("🍊 Orange Juice Demand Prediction")
    st.markdown("**MSIS 522 — Analytics and Machine Learning | Foster School of Business, UW**")

    st.markdown("---")

    st.header("📌 About the Dataset")
    st.markdown(f"""
This project uses the **Dominick's Finer Foods Orange Juice dataset**, a classic retail analytics dataset
widely used in marketing and demand-forecasting research. It contains **{len(df):,} weekly store-level
observations** across multiple retail locations, capturing how orange juice sales respond to pricing,
promotions, brand competition, and local demographics.

**Target variable:** `logmove` — the natural logarithm of the number of units sold in a given week.
Working in log-space stabilizes variance and makes the model less sensitive to extreme sales weeks.

**Key features include:**
- **`price`** — the shelf price of the orange juice product that week
- **`feat`** — a binary indicator of whether the product was featured in a store flyer or promotion (1 = yes, 0 = no)
- **`brand`** — the orange juice brand (e.g., Tropicana, Minute Maid, Dominick's own-label)
- **Demographic variables** — store-level characteristics such as average customer age (`AGE60`),
  household income (`INCOME`), and others describing the local customer base
""")

    st.header("💡 Why This Problem Matters")
    st.markdown("""
Accurate demand forecasting is a cornerstone of retail operations. Retailers and manufacturers use
sales predictions to make decisions about:

- **Inventory planning** — how much stock to order to avoid stockouts or overstock
- **Pricing strategy** — understanding price elasticity to set competitive prices without sacrificing margin
- **Promotional ROI** — determining whether a promotional feature ad actually drives incremental volume
- **Assortment decisions** — deciding which brands and sizes to carry in which stores

In the orange juice category specifically, demand is highly price-sensitive and heavily influenced by
promotions. A model that can accurately predict sales lift from a price cut or a feature ad is directly
actionable for category managers and supply chain planners.
""")

    st.header("🔬 Approach & Key Findings")
    st.markdown("""
This project implements the full data science workflow:

1. **Descriptive Analytics** — Exploring distributions, feature relationships, and the correlation structure
   of the dataset to build intuition before modeling.
2. **Predictive Modeling** — Seven models were trained and compared: Linear Regression, Ridge, Lasso,
   Decision Tree (CART), Random Forest, LightGBM, and a Neural Network (MLP).
3. **Explainability** — SHAP analysis on the best tree-based model reveals *which* features drive
   predictions and *in which direction*.
4. **Deployment** — All findings are surfaced in this interactive Streamlit app.

**Key findings:**
- 🏆 **LightGBM and Random Forest achieve the lowest RMSE**, significantly outperforming linear baselines.
  This confirms that demand is driven by nonlinear interactions (e.g., promotional effects differ by brand).
- 📉 **Price is the single strongest predictor** — higher prices are strongly associated with lower sales,
  consistent with standard demand theory.
- 📢 **Promotions (`feat`) have a large positive impact** — featured products see a substantial sales lift,
  and this effect is captured across all model types.
- 🏷️ **Brand identity matters** — even controlling for price, some brands consistently outsell others,
  likely reflecting consumer loyalty and brand equity.
""")

# ─────────────────────────────────────────
# TAB 2 — DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────

with tabs[1]:

    st.header("📊 Descriptive Analytics")
    st.markdown("""
Before building any model, it is essential to understand the data. The visualizations below explore
the distribution of the target variable, key feature relationships, and the overall correlation structure
of the dataset.
""")

    # ── Target Distribution ──
    st.subheader("1. Target Distribution: `logmove`")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["logmove"], kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of log(Units Sold)", fontsize=13)
    ax.set_xlabel("logmove (log of units sold)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.caption("""
**What this shows:** The distribution of `logmove` is approximately bell-shaped with a slight right skew,
centered around 9–10 (corresponding to roughly 8,000–22,000 units). Using log-scale sales as the target
variable reduces the influence of occasional blockbuster promotional weeks and makes residuals more
normally distributed — a desirable property for regression models.
""")

    # ── Brand vs Sales ──
    st.subheader("2. Sales by Brand")
    fig, ax = plt.subplots(figsize=(7, 4))
    brand_order = df.groupby("brand")["logmove"].median().sort_values(ascending=False).index
    sns.boxplot(x="brand", y="logmove", data=df, order=brand_order, palette="Set2", ax=ax)
    ax.set_title("Log Sales by Brand", fontsize=13)
    ax.set_xlabel("Brand")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** There are meaningful differences in median sales across brands. The spread of the
boxes also indicates varying week-to-week volatility by brand. This confirms that `brand` will be an
important predictor — models need to account for brand-level baseline differences in demand.
""")

    # ── Price vs Sales ──
    st.subheader("3. Price vs. Sales")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x="price", y="logmove", data=df.sample(min(2000, len(df)), random_state=42),
                    alpha=0.3, ax=ax, color="darkorange")
    ax.set_title("Price vs. Log Sales", fontsize=13)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** There is a clear negative relationship between price and sales — as price increases,
units sold decrease. This is consistent with standard demand theory and the concept of price elasticity.
The relationship appears somewhat nonlinear, which partly explains why tree-based models outperform
linear regression on this dataset.
""")

    # ── Promotion vs Sales ──
    st.subheader("4. Promotional Feature vs. Sales")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x="feat", y="logmove", data=df, palette=["#d9534f", "#5cb85c"], ax=ax)
    ax.set_title("Sales: Not Featured vs. Featured", fontsize=13)
    ax.set_xlabel("Featured in promotion (0 = No, 1 = Yes)")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** Products featured in store flyers or promotional displays (feat = 1) show substantially
higher median sales compared to non-featured weeks. This promotional lift is one of the strongest signals
in the dataset and will be a top SHAP feature. Retailers can use this insight to quantify the ROI of
running a promotion for a given brand.
""")

    # ── Pairplot ──
    st.subheader("5. Pairwise Feature Relationships")
    st.markdown("*Showing a random sample of 500 rows for readability.*")
    fig = sns.pairplot(
        df[["price", "logmove", "AGE60", "INCOME"]].sample(500, random_state=42),
        diag_kind="kde", plot_kws={"alpha": 0.3}
    )
    fig.fig.suptitle("Pairplot: Price, Sales, Age, Income", y=1.02, fontsize=12)
    st.pyplot(fig)
    st.caption("""
**What this shows:** The pairplot reveals relationships among the four most analytically interesting
continuous variables. `price` and `logmove` show the clearest negative correlation. `INCOME` and `AGE60`
show relatively weak direct correlations with sales, but they may interact with price sensitivity —
wealthier or older customer bases may be less price-elastic.
""")

    # ── Correlation Heatmap ──
    st.subheader("6. Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", mask=mask,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Lower Triangle)", fontsize=13)
    st.pyplot(fig)
    st.caption("""
**What this shows:** `price` has the strongest negative correlation with `logmove` among all numeric
features, confirming its role as the dominant driver of sales. `feat` (promotion) shows a moderate
positive correlation. Demographic variables like `INCOME` and `AGE60` show weaker correlations with
the target individually, but they likely contribute through interaction effects that tree-based models
can capture. Multicollinearity between some demographic features (if present) may inflate linear model
coefficients but does not affect tree-based models.
""")

# ─────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────

with tabs[2]:

    st.header("🤖 Model Performance Comparison")
    st.markdown("""
Seven models were trained and evaluated on a held-out **30% test set** (using `random_state=42`).
All tree-based models and the neural network were tuned via **5-fold cross-validation with GridSearchCV**
on the training set. The table and charts below summarize test-set performance.

**Metrics reported:**
- **RMSE** (Root Mean Squared Error) — penalizes large errors more heavily; lower is better
- **MAE** (Mean Absolute Error) — average absolute prediction error; lower is better
- **R²** (Coefficient of Determination) — proportion of variance explained; higher is better (max = 1.0)
""")

    if os.path.exists("model_results.csv"):
        results = pd.read_csv("model_results.csv")

        st.subheader("📋 Results Table")
        st.dataframe(results.style.highlight_min(subset=["RMSE", "MAE"], color="#c6efce")
                                  .highlight_max(subset=["R2"], color="#c6efce"),
                     use_container_width=True)

        st.subheader("📊 RMSE Comparison (lower = better)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_results = results.sort_values("RMSE")
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sorted_results))]
        sns.barplot(data=sorted_results, x="Model", y="RMSE", palette=colors, ax=ax)
        ax.set_title("Test-Set RMSE by Model", fontsize=13)
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig)
        st.caption("""
**The green bar marks the best-performing model.** LightGBM and Random Forest typically achieve the
lowest RMSE, demonstrating that capturing nonlinear feature interactions improves demand prediction
substantially over linear baselines.
""")

    else:
        st.warning("⚠️ `model_results.csv` not found. Please run `train_models.py` to generate results.")

    st.subheader("🧠 Model Discussion")
    st.markdown("""
| Model | Key Strength | Key Limitation |
|---|---|---|
| **Linear / Ridge / Lasso** | Fast, interpretable, good baseline | Cannot capture nonlinear interactions |
| **Decision Tree (CART)** | Fully interpretable, visualizable | Prone to overfitting; high variance |
| **Random Forest** | Robust, handles interactions, low variance | Slower to train; less interpretable |
| **LightGBM** | State-of-the-art accuracy, fast training | Many hyperparameters; slight black-box |
| **Neural Network (MLP)** | Can learn complex patterns | Requires scaling; needs more data to shine |

**Overall conclusion:** The tree-based ensemble models (Random Forest and LightGBM) outperform
linear and single-tree models because orange juice demand is driven by nonlinear interactions —
for example, the promotional lift may differ by brand, and price sensitivity may vary by store demographics.
The Neural Network performs competitively but requires careful tuning to match the gradient-boosted trees.
""")

# ─────────────────────────────────────────
# TAB 4 — EXPLAINABILITY & PREDICTION
# ─────────────────────────────────────────

with tabs[3]:

    st.header("🔍 Model Explainability (SHAP)")
    st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a game-theoretic framework for explaining individual model
predictions. Each feature is assigned a SHAP value representing its contribution to pushing the prediction
above or below the model's baseline (average) prediction.

- **Positive SHAP value** → this feature increased the predicted sales for this observation
- **Negative SHAP value** → this feature decreased the predicted sales

The plots below use the **Random Forest** model, which has strong predictive performance and
supports efficient SHAP computation.
""")

    # ── Train a fresh RF purely from the CSV for SHAP ────────────────────────
    # The saved .pkl pipelines contain a SimpleImputer pickled with an older
    # sklearn. Any call to pipeline.predict() or pipeline.transform() internally
    # runs that broken imputer and raises:
    #   AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
    #
    # Fix: never load the pkl for SHAP at all. Instead, train a lightweight
    # RandomForest directly on the CSV data using only pandas + current sklearn.
    # This model is used exclusively for SHAP — the pkl models are still used
    # for predictions in the interactive section below.
    # ─────────────────────────────────────────────────────────────────────────

    @st.cache_resource
    def build_shap_model(_df):
        """
        Trains a fresh RandomForestRegressor on the already-loaded DataFrame.
        No Pipeline, no saved imputer — pure current-sklearn operations only.
        Cached so it only trains once per Streamlit session.
        _df is prefixed with _ so Streamlit doesn't try to hash it.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        data = _df.copy()
        X = data.drop(columns=["logmove"]).copy()
        y = data["logmove"].copy()

        # Encode categoricals with label encoding (no imputer needed — oj.csv has no nulls)
        cat_cols_local = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols_local:
            X[col] = X[col].astype("category").cat.codes

        # Fill any remaining NaNs with column median (pure pandas, no sklearn imputer)
        X = X.fillna(X.median(numeric_only=True))
        X = X.astype(float)

        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        return rf, X.columns.tolist(), X

    with st.spinner("Training SHAP model from data (first load only)..."):
        shap_rf, shap_feature_names, X_all_enc = build_shap_model(df)

    X_explain = X_all_enc.sample(200, random_state=42)

    with st.spinner("Computing SHAP values with TreeExplainer..."):
        try:
            explainer = shap.TreeExplainer(shap_rf)
            shap_vals = explainer.shap_values(X_explain)
            shap_ok   = True
        except Exception as e:
            shap_ok = False
            st.error(f"SHAP computation failed: {e}")

    if shap_ok:
        st.subheader("1. SHAP Summary Plot (Beeswarm)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X_explain, feature_names=shap_feature_names, show=False)
        st.pyplot(fig)
        st.caption("""
**How to read this:** Each dot is one observation. Its x-position shows how much that feature
shifted the prediction up or down from the baseline average. Color shows the raw feature value
(red = high, blue = low). Features are sorted by mean absolute SHAP value — the most influential
drivers appear at the top.
""")

        st.subheader("2. Mean Absolute SHAP Feature Importance")
        mean_shap = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature":    shap_feature_names,
            "Mean |SHAP|": mean_shap
        }).sort_values("Mean |SHAP|", ascending=True)

        fig, ax = plt.subplots(figsize=(7, max(3, len(shap_feature_names) * 0.35)))
        ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="steelblue")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("Global Feature Importance (SHAP — Random Forest)")
        st.pyplot(fig)
        st.caption("""
**How to read this:** Each bar is the average magnitude of a feature's SHAP contribution across
200 observations. Longer bars mean larger average prediction impact. Unlike tree impurity importance,
SHAP values reflect actual prediction contributions and are not biased toward high-cardinality features.
""")

    st.markdown("---")

    st.markdown("---")
    st.header("🎯 Interactive Prediction")
    st.markdown("""
Use the controls below to set feature values and see what any model predicts for weekly orange
juice sales. All other features are held at their dataset mean (numeric) or most common value
(categorical). The predicted value is in **log-units** — to convert to actual units: `exp(value)`.

> **Note:** Models are trained fresh from the dataset on first load and cached for the session.
> This avoids version-compatibility issues with saved model files.
""")

    # ── Train all models fresh from data (cached per session) ────────────────
    # We do NOT load the .pkl files here. The saved pipelines contain a
    # SimpleImputer pickled under an older sklearn version which raises:
    #   AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
    # Training fresh from the CSV costs ~15s once then is cached for the session.
    # ─────────────────────────────────────────────────────────────────────────

    @st.cache_resource
    def build_all_models(_df):
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split

        data = _df.copy()
        X = data.drop(columns=["logmove"]).copy()
        y = data["logmove"].copy()

        cat_cols_l = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols_l:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(X.median(numeric_only=True)).astype(float)

        feat_names = X.columns.tolist()
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=42)

        specs = {
            "Linear Regression":    LinearRegression(),
            "Ridge Regression":     Ridge(),
            "Lasso Regression":     Lasso(),
            "Decision Tree (CART)": DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest":        RandomForestRegressor(n_estimators=100, max_depth=8,
                                                          random_state=42, n_jobs=-1),
            "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(128, 128),
                                                  max_iter=300, random_state=42),
        }
        try:
            from lightgbm import LGBMRegressor
            specs["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        except ImportError:
            pass

        trained = {}
        for label, mdl in specs.items():
            mdl.fit(X_tr, y_tr)
            trained[label] = mdl

        return trained, feat_names

    with st.spinner("Training models from data (first load only — ~15 seconds)..."):
        all_models, pred_feature_names = build_all_models(df)

    selected_label = st.selectbox(
        "**Select a model to use for prediction:**",
        list(all_models.keys())
    )
    chosen_model = all_models[selected_label]

    col1, col2 = st.columns(2)
    with col1:
        price_val = st.slider(
            "💲 Price ($)", 0.5, 5.0, float(df["price"].mean()), step=0.05,
            help="The retail shelf price of the orange juice this week."
        )
    with col2:
        feat_val = st.selectbox(
            "📢 On Promotion? (feat)", [0, 1],
            format_func=lambda x: "Yes — featured in flyer" if x == 1 else "No promotion",
            help="Whether the product was featured in a store advertisement this week."
        )

    # Build encoded input row — no sklearn imputer, pure pandas
    X_input = pd.DataFrame([{
        col: df[col].mean() if col in NUM_COLS else df[col].mode()[0]
        for col in pred_feature_names
    }])
    if "price" in X_input.columns:
        X_input["price"] = price_val
    if "feat" in X_input.columns:
        X_input["feat"] = float(feat_val)
    for col in CAT_COLS:
        if col in X_input.columns:
            X_input[col] = pd.Categorical(
                X_input[col], categories=df[col].astype("category").cat.categories
            ).codes
    X_input = X_input[pred_feature_names].astype(float)

    prediction      = chosen_model.predict(X_input)[0]
    predicted_units = int(np.exp(prediction))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("📦 Predicted log(sales)", f"{prediction:.3f}")
    col_b.metric("📦 Predicted units sold (approx.)", f"{predicted_units:,}")

    st.info(f"""
**Interpretation:** With a price of **${price_val:.2f}** and promotion status
**{'ON' if feat_val == 1 else 'OFF'}**, the **{selected_label}** model predicts approximately
**{predicted_units:,} units** will be sold (log-sales = {prediction:.3f}).
All other features are held at their dataset averages.
""")

    # SHAP waterfall (RF only — uses the same fresh model, no pkl)
    if selected_label == "Random Forest":
        st.subheader("🌊 SHAP Waterfall Plot for Your Input")
        st.markdown("""
The waterfall plot below explains *this specific prediction* — showing how each feature pushed
the predicted sales up (red ↑) or down (blue ↓) from the model's average baseline.
""")
        try:
            exp_wf   = shap.TreeExplainer(chosen_model)
            sv_wf    = exp_wf.shap_values(X_input)
            sv_row   = sv_wf[0]
            base_val = exp_wf.expected_value

            top_n   = min(10, len(pred_feature_names))
            indices = np.argsort(np.abs(sv_row))[::-1][:top_n]
            labels  = [pred_feature_names[i] for i in indices][::-1]
            values  = sv_row[indices][::-1]
            colors  = ["#e74c3c" if v > 0 else "#3498db" for v in values]

            fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.45)))
            ax.barh(labels, values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on log sales)")
            ax.set_title(
                f"Waterfall — baseline: {base_val:.3f} | "
                f"prediction: {base_val + sv_row.sum():.3f}"
            )
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not generate waterfall plot: {e}")
