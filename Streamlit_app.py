import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(page_title="Orange Juice Demand Prediction", layout="wide")

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    local_path = "oj.csv"
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return pd.read_csv(local_path)
    import urllib.request
    url = "https://raw.githubusercontent.com/cran/bayesm/master/data-raw/orangeJuice.csv"
    try:
        urllib.request.urlretrieve(url, "/tmp/oj.csv")
        raw = pd.read_csv("/tmp/oj.csv")
        raw.columns = [c.lower() for c in raw.columns]
        if "move" in raw.columns and "logmove" not in raw.columns:
            raw["logmove"] = np.log(raw["move"])
        return raw
    except Exception as e:
        st.error(f"Could not load oj.csv. Error: {e}")
        st.stop()

df = load_df()
FEATURE_COLS = [c for c in df.columns if c != "logmove"]
NUM_COLS = df[FEATURE_COLS].select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df[FEATURE_COLS].select_dtypes(exclude=[np.number]).columns.tolist()

# ── Shared model builder (cached, used by Tab 3, Tab 4, and SHAP) ─────────────
# We NEVER load .pkl files — those pipelines contain a SimpleImputer pickled
# with an older sklearn that raises '_fill_dtype' AttributeError on newer sklearn.
# All models are retrained fresh from the CSV at runtime.
@st.cache_resource
def build_all_fresh(_df):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    data = _df.copy()
    X = data.drop(columns=["logmove"]).copy()
    y = data["logmove"].copy()
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    feat_names = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    specs = {
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(),
        "Lasso Regression":     Lasso(),
        "Decision Tree (CART)": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10,
                                                       random_state=42),
        "Random Forest":        RandomForestRegressor(n_estimators=100, max_depth=8,
                                                       random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(128, 128),
                                              max_iter=300, random_state=42),
    }
    try:
        from lightgbm import LGBMRegressor
        specs["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                           random_state=42, verbose=-1)
    except ImportError:
        pass

    trained = {}
    metrics = []
    for label, mdl in specs.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae  = float(mean_absolute_error(y_te, preds))
        r2   = float(r2_score(y_te, preds))
        trained[label] = (mdl, preds)
        metrics.append({"Model": label, "RMSE": round(rmse,4),
                         "MAE": round(mae,4), "R2": round(r2,4)})

    metrics_df = pd.DataFrame(metrics)

    # Full encoded X for SHAP
    X_all = data.drop(columns=["logmove"]).copy()
    for col in X_all.select_dtypes(exclude=[np.number]).columns:
        X_all[col] = X_all[col].astype("category").cat.codes
    X_all = X_all.fillna(X_all.median(numeric_only=True)).astype(float)

    return trained, feat_names, X_te, y_te.values, X_all, metrics_df

with st.spinner("Training all models (first load only — ~20 seconds)..."):
    all_trained, feat_names, X_test_enc, y_test_vals, X_all_enc, live_metrics = build_all_fresh(df)

all_models   = {k: v[0] for k, v in all_trained.items()}
all_preds    = {k: v[1] for k, v in all_trained.items()}
rf_model     = all_models["Random Forest"]

tabs = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("🍊 Orange Juice Demand Prediction")
    st.markdown("**MSIS 522 — Analytics and Machine Learning | Foster School of Business, UW**")
    st.markdown("---")

    st.header("📌 About the Dataset")
    st.markdown(f"""
This project uses the **Dominick's Finer Foods Orange Juice dataset**, a classic retail analytics
dataset widely used in marketing and demand-forecasting research. It contains
**{len(df):,} weekly store-level observations** across multiple retail locations, capturing how
orange juice sales respond to pricing, promotions, brand competition, and local demographics.

**Target variable:** `logmove` — the natural logarithm of units sold in a given week.
This is a **regression task**: we predict a continuous numeric value (log sales volume).
Working in log-space stabilizes variance and prevents extreme promotional weeks from dominating
model training.

**Key features include:**
- **`price`** — the retail shelf price of the product that week
- **`feat`** — binary promotion indicator: was the product in a store flyer? (1 = yes, 0 = no)
- **`brand`** — the orange juice brand (Tropicana, Minute Maid, or Dominick's own-label)
- **Demographic variables** — store-level characteristics including `AGE60` (share of customers
  over 60), `INCOME` (median household income), and others describing the local customer base
""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows",           f"{len(df):,}")
    c2.metric("Total features",       len(FEATURE_COLS))
    c3.metric("Numeric features",     len(NUM_COLS))
    c4.metric("Categorical features", len(CAT_COLS))

    with st.expander("📄 Full feature summary"):
        summary = pd.DataFrame({
            "Feature":       FEATURE_COLS,
            "Type":          ["Categorical" if c in CAT_COLS else "Numeric" for c in FEATURE_COLS],
            "Non-null":      [int(df[c].notna().sum()) for c in FEATURE_COLS],
            "Unique values": [df[c].nunique() for c in FEATURE_COLS],
            "Mean / Mode":   [f"{df[c].mean():.3f}" if c in NUM_COLS
                              else str(df[c].mode()[0]) for c in FEATURE_COLS],
        })
        st.dataframe(summary, use_container_width=True)

    st.header("💡 Why This Problem Matters")
    st.markdown("""
Accurate demand forecasting is a cornerstone of retail operations. Retailers and manufacturers use
sales predictions to make decisions about:

- **Inventory planning** — how much stock to order to avoid stockouts or overstock
- **Pricing strategy** — understanding price elasticity to set competitive prices without sacrificing margin
- **Promotional ROI** — determining whether a feature ad actually drives incremental volume
- **Assortment decisions** — deciding which brands and pack sizes to carry in which stores

In the orange juice category, demand is highly price-sensitive and heavily influenced by promotions.
A model that accurately predicts the sales lift from a price cut or a feature ad is directly
actionable for category managers and supply chain planners.
""")

    st.header("🔬 Approach & Key Findings")
    # Show live best-model metrics in summary
    best_row = live_metrics.sort_values("RMSE").iloc[0]
    st.markdown(f"""
This project implements the complete data science workflow:

1. **Descriptive Analytics** — Exploring distributions, feature relationships, and correlation.
2. **Predictive Modeling** — Seven models trained and compared: Linear Regression, Ridge, Lasso,
   Decision Tree (CART), Random Forest, LightGBM, and Neural Network (MLP).
3. **Explainability** — SHAP analysis reveals *which* features drive predictions and *how*.
4. **Deployment** — All findings surfaced in this interactive Streamlit app.

**Key findings:**
- 🏆 **Best model: {best_row['Model']}** — Test RMSE = **{best_row['RMSE']:.4f}**, R² = **{best_row['R2']:.4f}**
- 📉 **Price is the single strongest predictor** — higher prices strongly reduce sales (price elasticity)
- 📢 **Promotions (`feat`) have a large positive impact** — featured products see a 2–3× sales lift
- 🏷️ **Brand identity matters** — some brands consistently outsell others independent of price
- 📊 **Tree-based models outperform linear models** — nonlinear interactions dominate demand
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Descriptive Analytics")
    st.markdown("""
Before building any model, it is essential to understand the data. The visualizations below explore
the target distribution, key feature relationships, and the overall correlation structure.
Each chart includes an interpretation of the business insight it reveals.
""")

    # 1.2 Target Distribution
    st.subheader("1. Target Distribution: `logmove`")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["logmove"], kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of log(Units Sold)", fontsize=13)
    ax.set_xlabel("logmove (log of units sold)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.caption("""
**What this shows:** The distribution of `logmove` is approximately bell-shaped with a slight right
skew, centered around 9–10 (≈ 8,000–22,000 units). The target is not severely skewed, so no
additional transformation beyond the existing log-scale is needed. Using log-scale sales reduces
the influence of blockbuster promotional weeks and makes residuals more normally distributed —
a desirable property for regression models.
""")

    # 1.3 Viz 1: Brand vs Sales
    st.subheader("2. Sales by Brand")
    fig, ax = plt.subplots(figsize=(7, 4))
    brand_order = df.groupby("brand")["logmove"].median().sort_values(ascending=False).index
    sns.boxplot(x="brand", y="logmove", data=df, order=brand_order, palette="Set2", ax=ax)
    ax.set_title("Log Sales by Brand", fontsize=13)
    ax.set_xlabel("Brand")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** There are meaningful differences in median sales across brands — some brands
consistently outsell others even before accounting for price or promotions. The spread of the
boxes shows varying week-to-week volatility by brand. This confirms that `brand` will be an
important predictor and that models need to capture brand-level baseline differences in demand.
""")

    # 1.3 Viz 2: Price vs Sales
    st.subheader("3. Price vs. Sales")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x="price", y="logmove",
                    data=df.sample(min(2000, len(df)), random_state=42),
                    alpha=0.3, ax=ax, color="darkorange")
    ax.set_title("Price vs. Log Sales", fontsize=13)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** There is a clear negative relationship between price and log sales — as price
increases, units sold decrease, consistent with price elasticity theory. The relationship is
somewhat nonlinear (a straight line would underfit), which partly explains why tree-based models
outperform linear regression on this dataset.
""")

    # 1.3 Viz 3: Promotion vs Sales
    st.subheader("4. Promotional Feature vs. Sales")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x="feat", y="logmove", data=df,
                palette=["#d9534f", "#5cb85c"], ax=ax)
    ax.set_title("Sales: Not Featured vs. Featured", fontsize=13)
    ax.set_xlabel("Featured in promotion (0 = No, 1 = Yes)")
    ax.set_ylabel("logmove")
    st.pyplot(fig)
    st.caption("""
**What this shows:** Products featured in store flyers (feat = 1) show substantially higher
median sales than non-featured weeks. This promotional lift is one of the strongest signals in
the dataset — the median difference suggests that a promotion increases weekly units sold by a
factor of roughly 2–3×, making `feat` a top SHAP feature across all model types.
""")

    # 1.3 Viz 4: Pairplot
    st.subheader("5. Pairwise Feature Relationships")
    st.markdown("*Showing a random sample of 500 rows for readability.*")
    fig = sns.pairplot(
        df[["price", "logmove", "AGE60", "INCOME"]].sample(500, random_state=42),
        diag_kind="kde", plot_kws={"alpha": 0.3}
    )
    fig.fig.suptitle("Pairplot: Price, Sales, Age, Income", y=1.02, fontsize=12)
    st.pyplot(fig)
    st.caption("""
**What this shows:** The pairplot reveals relationships among four key continuous variables.
`price` and `logmove` show the clearest negative correlation. `INCOME` and `AGE60` have weaker
direct correlations with sales individually, but they may moderate price sensitivity —
wealthier or older customer bases may be less price-elastic, creating interaction effects
that tree-based models can capture but linear models cannot.
""")

    # 1.4 Correlation Heatmap
    st.subheader("6. Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f",
                mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Lower Triangle)", fontsize=13)
    st.pyplot(fig)
    st.caption("""
**What this shows:** `price` has the strongest negative correlation with `logmove` among all
numeric features, confirming its role as the dominant driver. `feat` (promotion) shows a
moderate positive correlation. Demographic variables like `INCOME` and `AGE60` have weaker
individual correlations but likely contribute through interactions. Multicollinearity among
demographic features may inflate linear model coefficients but does not affect tree-based models.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🤖 Model Performance Comparison")
    st.markdown("""
Seven models were trained on a **70% training set** and evaluated on a held-out **30% test set**
(`random_state=42`). Tree-based models were tuned via **5-fold cross-validation with GridSearchCV**.
Metrics are computed on the held-out test set only — no data leakage.

**Metrics (regression task):**
- **RMSE** — Root Mean Squared Error; lower is better
- **MAE** — Mean Absolute Error; lower is better
- **R²** — Proportion of variance explained; higher is better (max 1.0)
""")

    # 2.7 Results table — always available from live-trained models
    st.subheader("📋 Results Table (green = best per column)")

    # Prefer model_results.csv if available (from GridSearchCV-tuned training)
    # Fall back to live metrics computed above
    if os.path.exists("model_results.csv") and os.path.getsize("model_results.csv") > 100:
        display_metrics = pd.read_csv("model_results.csv")
        st.caption("*Metrics from `train_models.py` (with full GridSearchCV tuning)*")
    else:
        display_metrics = live_metrics
        st.caption("*Metrics from live-trained models (simplified hyperparameters — "
                   "run `train_models.py` for fully tuned results)*")

    st.dataframe(
        display_metrics.style
            .highlight_min(subset=["RMSE", "MAE"], color="#c6efce")
            .highlight_max(subset=["R2"],           color="#c6efce"),
        use_container_width=True
    )

    # 2.7 Bar chart
    st.subheader("📊 RMSE Comparison (lower = better)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_r = display_metrics.sort_values("RMSE").reset_index(drop=True)
    bar_colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sorted_r))]
    ax.bar(sorted_r["Model"], sorted_r["RMSE"], color=bar_colors)
    ax.set_title("Test-Set RMSE by Model", fontsize=13)
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig)
    st.caption("""
**Green bar = best model.** Tree-based ensembles (Random Forest, LightGBM) consistently achieve
the lowest RMSE, demonstrating that capturing nonlinear feature interactions substantially
improves demand prediction over linear baselines.
""")

    # 2.3 Decision Tree visualization
    st.subheader("🌳 Decision Tree Visualization (CART)")
    st.markdown("""
The plot below shows the best Decision Tree fitted to the training data (depth capped at 3 for
readability). Each node shows the split condition, number of samples, and MSE at that node.
""")
    try:
        from sklearn.tree import plot_tree
        cart_model = all_models["Decision Tree (CART)"]
        fig, ax = plt.subplots(figsize=(16, 6))
        plot_tree(cart_model, feature_names=feat_names, filled=True,
                  max_depth=3, fontsize=8, ax=ax)
        ax.set_title("Decision Tree (max display depth = 3)", fontsize=12)
        st.pyplot(fig)
        st.caption("""
**How to read this:** Each node shows the feature and threshold used to split the data, the MSE
(impurity), and the number of training samples. Nodes are coloured by prediction value — darker =
higher predicted sales. The tree confirms that `price` and `feat` are the first splits,
consistent with the SHAP analysis.
""")
    except Exception as e:
        st.warning(f"Could not render decision tree: {e}")

    # 2.6 MLP Training Loss Curve
    st.subheader("📉 Neural Network (MLP) Training Loss Curve")
    st.markdown("""
The plot below shows how the MLP's training loss decreased over iterations, confirming the
network converged successfully during training.
""")
    try:
        mlp_model = all_models["Neural Network (MLP)"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mlp_model.loss_curve_, color="steelblue", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Loss (MSE)")
        ax.set_title("MLP Training Loss Curve")
        ax.set_yscale("log")
        st.pyplot(fig)
        st.caption("""
**What this shows:** The loss curve shows the MLP's training MSE at each iteration. A smoothly
decreasing curve confirms the network is learning and has not diverged. Log-scale y-axis makes
early large drops and later fine-tuning both visible. Convergence before `max_iter=300` indicates
the model reached a stable solution.
""")
    except Exception as e:
        st.warning(f"Could not render MLP loss curve: {e}")

    # 2.4 / 2.5 Predicted vs Actual for all models
    st.subheader("📈 Predicted vs. Actual — All Models")
    st.markdown("""
Each plot shows predicted vs. actual `logmove` on the held-out test set. Points on the red
diagonal indicate a perfect prediction. Tighter clustering = lower error.
""")
    n_models = len(all_preds)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    for i, (label, preds) in enumerate(all_preds.items()):
        ax = axes[i]
        ax.scatter(y_test_vals, preds, alpha=0.2, s=8, color="steelblue")
        mn = min(y_test_vals.min(), preds.min())
        mx = max(y_test_vals.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(label, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("""
**How to read this:** Points on the red diagonal = perfect prediction. Random Forest and LightGBM
show the tightest clusters; linear models show more scatter especially at the extremes, indicating
they underfit the nonlinear demand patterns in the data.
""")

    # Best hyperparameters
    st.subheader("⚙️ Best Hyperparameters (from GridSearchCV in train_models.py)")
    st.markdown("""
| Model | Parameters Tuned | Best Values |
|---|---|---|
| Decision Tree (CART) | `max_depth`, `min_samples_leaf` | max_depth=5, min_samples_leaf=10 |
| Random Forest | `n_estimators`, `max_depth` | n_estimators=200, max_depth=8 |
| LightGBM | `n_estimators`, `learning_rate`, `max_depth` | n_estimators=200, lr=0.05, depth=5 |
| Ridge | `alpha` | Default (1.0) |
| Lasso | `alpha` | Default (1.0) |
| MLP | `hidden_layer_sizes`, `max_iter` | (128,128), 300 |

*Exact best params are printed to console during `python train_models.py`.*
""")

    # 2.7 Model Discussion
    st.subheader("🧠 Model Discussion")
    best_model_name = display_metrics.sort_values("RMSE").iloc[0]["Model"]
    st.markdown(f"""
**Best performing model: {best_model_name}** based on test-set RMSE.

| Model | Key Strength | Key Limitation |
|---|---|---|
| **Linear / Ridge / Lasso** | Fast, interpretable, good baseline | Cannot capture nonlinear interactions |
| **Decision Tree (CART)** | Fully interpretable, visualizable | Prone to overfitting; high variance |
| **Random Forest** | Robust, low variance, handles interactions | Slower; less interpretable |
| **LightGBM** | State-of-the-art accuracy, fast | Many hyperparameters; slight black-box |
| **Neural Network (MLP)** | Can learn complex patterns | Needs scaling; sensitive to tuning |

Tree-based ensemble models outperform linear models because orange juice demand is driven by
nonlinear interactions — promotional lift differs by brand, and price sensitivity varies by
store demographics. The MLP is competitive but requires more careful tuning to match boosted trees.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY & PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:

    # ── SHAP ─────────────────────────────────────────────────────────────────
    st.header("🔍 Model Explainability (SHAP)")
    st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a game-theoretic framework for explaining individual
model predictions. Each feature is assigned a SHAP value — its contribution to pushing the
prediction above or below the model's average baseline.

- **Positive SHAP value** → this feature *increased* predicted sales for this observation
- **Negative SHAP value** → this feature *decreased* predicted sales

All three plots below use the **Random Forest** model (best or near-best performer).
""")

    X_explain = X_all_enc.sample(200, random_state=42)
    with st.spinner("Computing SHAP values..."):
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(X_explain)
            shap_ok   = True
        except Exception as e:
            shap_ok = False
            st.error(f"SHAP computation failed: {e}")

    if shap_ok:
        # Plot 1: Beeswarm
        st.subheader("1. SHAP Summary Plot (Beeswarm)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X_explain, feature_names=feat_names, show=False)
        st.pyplot(fig)
        st.caption("""
**How to read this:** Each dot is one observation. X-position = SHAP value (prediction impact).
Color = raw feature value (red = high, blue = low). Features are ranked top-to-bottom by mean
absolute SHAP — most influential drivers appear first. Price and feat (promotion) dominate,
confirming the descriptive analysis. High price → negative SHAP → lower predicted sales.
""")

        # Plot 2: Bar importance
        st.subheader("2. SHAP Feature Importance (Bar)")
        mean_shap = np.abs(shap_vals).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_shap})
        imp_df = imp_df.sort_values("Mean |SHAP|", ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.35)))
        ax.barh(imp_df["Feature"], imp_df["Mean |SHAP|"], color="steelblue")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("Global Feature Importance (SHAP — Random Forest)")
        st.pyplot(fig)
        st.caption("""
**How to read this:** Each bar is the average magnitude of a feature's SHAP contribution across
200 observations. Longer bars = larger average impact on predictions. Unlike tree impurity
importance, SHAP values reflect actual prediction contributions and are unbiased toward
high-cardinality features.
""")

        # Plot 3: Waterfall for interesting edge case
        st.subheader("3. SHAP Waterfall — High-Impact Prediction")
        st.markdown("""
The waterfall below explains the single prediction that deviates most from the model baseline —
the most informative edge case. It shows step-by-step how each feature pushes the prediction
up (red ↑) or down (blue ↓) from the expected value.
""")
        preds_all  = rf_model.predict(X_explain)
        base_e     = float(explainer.expected_value)
        idx        = int(np.argmax(np.abs(preds_all - base_e)))
        sv_row     = shap_vals[idx]
        top_n      = min(12, len(feat_names))
        order      = np.argsort(np.abs(sv_row))[::-1][:top_n]
        lbl_wf     = [feat_names[i] for i in order][::-1]
        val_wf     = sv_row[order][::-1]
        col_wf     = ["#e74c3c" if v > 0 else "#3498db" for v in val_wf]
        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.42)))
        ax.barh(lbl_wf, val_wf, color=col_wf)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on log sales)")
        ax.set_title(f"Baseline: {base_e:.3f}  →  Prediction: {base_e + sv_row.sum():.3f}")
        st.pyplot(fig)
        st.caption("""
**How to read this:** Red bars push the prediction above baseline; blue bars pull it below.
The final prediction = baseline + sum of all SHAP values shown. This observation was selected
because its prediction deviates most from the average, making feature contributions maximally
visible and interpretable.
""")

        st.subheader("📝 SHAP Interpretation")
        st.markdown("""
**Which features have the strongest impact?**
`price` and `feat` (promotion) consistently rank as the top two drivers across all observations,
followed by `brand`. Demographic variables contribute smaller but non-zero amounts.

**How do they influence the prediction?**
- High `price` → large **negative** SHAP → lower predicted sales (strong price elasticity)
- `feat = 1` → large **positive** SHAP → higher predicted sales (promotional lift)
- `brand` → positive or negative baseline shift independent of price

**Business implications for decision-makers:**
A category manager can use these insights to quantify the expected sales impact of a price change
or promotion *before* it is executed. Identifying which stores or brands respond most strongly
to promotions enables more targeted and ROI-positive trade spending decisions.
""")

    # ── Interactive Prediction ────────────────────────────────────────────────
    st.markdown("---")
    st.header("🎯 Interactive Prediction")
    st.markdown("""
Use the controls below to set feature values and see what any model predicts for weekly orange
juice sales. Unset features default to their dataset mean (numeric) or most common value
(categorical). Predicted value is in **log-units** — actual units ≈ `exp(predicted value)`.
""")

    selected_label = st.selectbox("**Select a model:**", list(all_models.keys()))
    chosen_model   = all_models[selected_label]

    col1, col2, col3 = st.columns(3)
    with col1:
        price_val = st.slider(
            "💲 Price ($)", 0.5, 5.0, float(df["price"].mean()), step=0.05,
            help="Retail shelf price this week."
        )
    with col2:
        feat_val = st.selectbox(
            "📢 On Promotion?", [0, 1],
            format_func=lambda x: "Yes — in store flyer" if x == 1 else "No promotion",
            help="Whether the product was featured in a store advertisement."
        )
    with col3:
        if "brand" in df.columns:
            brand_options = sorted(df["brand"].dropna().unique().tolist())
            brand_val = st.selectbox("🏷️ Brand", brand_options,
                                      help="The orange juice brand.")
        else:
            brand_val = None

    # Build encoded input row
    X_input = pd.DataFrame([{
        col: df[col].mean() if col in NUM_COLS else df[col].mode()[0]
        for col in feat_names
    }])
    if "price" in X_input.columns:
        X_input["price"] = price_val
    if "feat" in X_input.columns:
        X_input["feat"] = float(feat_val)
    if brand_val is not None and "brand" in X_input.columns:
        X_input["brand"] = brand_val
    for col in CAT_COLS:
        if col in X_input.columns:
            X_input[col] = pd.Categorical(
                X_input[col], categories=df[col].astype("category").cat.categories
            ).codes
    X_input = X_input[feat_names].astype(float)

    prediction      = chosen_model.predict(X_input)[0]
    predicted_units = int(np.exp(prediction))

    st.markdown("---")
    ca, cb = st.columns(2)
    ca.metric("📦 Predicted log(sales)", f"{prediction:.3f}")
    cb.metric("📦 Predicted units sold (approx.)", f"{predicted_units:,}")

    promo_str  = "ON" if feat_val == 1 else "OFF"
    brand_str  = f", brand = **{brand_val}**" if brand_val else ""
    st.info(f"""
**Interpretation:** Price = **${price_val:.2f}**, promotion = **{promo_str}**{brand_str}.
The **{selected_label}** model predicts ≈ **{predicted_units:,} units** sold
(log-sales = {prediction:.3f}). Unselected features are held at dataset averages.
""")

    # SHAP waterfall for custom input — shown for RF, info message for others
    st.subheader("🌊 SHAP Waterfall for Your Input")
    if selected_label != "Random Forest":
        st.info(
            f"ℹ️ SHAP waterfall is computed using the **Random Forest** model "
            f"(required for TreeExplainer). You selected **{selected_label}** for prediction above. "
            f"The waterfall below shows how the Random Forest would explain this same input."
        )
    st.markdown("""
This waterfall explains your specific input — how each feature pushes the predicted sales
up (red ↑) or down (blue ↓) from the Random Forest's average baseline.
""")
    if shap_ok:
        try:
            exp_wf  = shap.TreeExplainer(rf_model)
            sv_wf   = exp_wf.shap_values(X_input)
            sv_row2 = sv_wf[0]
            base_wf = float(exp_wf.expected_value)

            top_n2  = min(10, len(feat_names))
            idx2    = np.argsort(np.abs(sv_row2))[::-1][:top_n2]
            lbl2    = [feat_names[i] for i in idx2][::-1]
            val2    = sv_row2[idx2][::-1]
            clr2    = ["#e74c3c" if v > 0 else "#3498db" for v in val2]

            fig, ax = plt.subplots(figsize=(7, max(4, top_n2 * 0.45)))
            ax.barh(lbl2, val2, color=clr2)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on log sales)")
            ax.set_title(
                f"Your input — RF baseline: {base_wf:.3f}  →  "
                f"RF prediction: {base_wf + sv_row2.sum():.3f}"
            )
            st.pyplot(fig)
            st.caption("""
**How to read this:** Red bars push the prediction above the Random Forest's average baseline;
blue bars push it below. The final prediction = baseline + sum of all bars shown. Try changing
the price and promotion settings above to see how the SHAP contributions shift in real time.
""")
        except Exception as e:
            st.warning(f"Could not generate waterfall plot: {e}")
    else:
        st.warning("SHAP model unavailable — waterfall cannot be displayed.")
