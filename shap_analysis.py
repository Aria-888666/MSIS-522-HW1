"""
shap_analysis.py
----------------
Generates SHAP plots for the best-performing tree model (Random Forest).

KEY DESIGN DECISION — why KernelExplainer instead of TreeExplainer:
  All models are saved as full sklearn Pipelines (preprocessor + model).
  Calling model.named_steps["prep"].transform() directly on a pickled pipeline
  can fail with "_fill_dtype" AttributeError when the sklearn version at runtime
  differs from the version used to train and pickle the pipeline.

  KernelExplainer treats the pipeline as a pure black box — it only ever calls
  pipeline.predict(X_raw), so it never touches internal pipeline attributes.
  This makes it robust to sklearn version mismatches.

  Tradeoff: KernelExplainer is slower than TreeExplainer, which is why we use
  a small background sample (50 rows) and a small explanation set (100 rows).
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_data, split_data

# ── Load data and model ───────────────────────────────────────────────────────
df = load_data()
X_train, X_test, y_train, y_test, preprocessor = split_data(df)

pipeline = joblib.load("models/random_forest.pkl")

FEATURE_COLS = X_test.columns.tolist()

# ── Encode categoricals for SHAP input ───────────────────────────────────────
# KernelExplainer passes arrays through pipeline.predict, which handles all
# internal preprocessing. We label-encode categoricals here so the array is
# fully numeric (avoids pandas object-dtype issues in shap internals).
cat_cols = X_test.select_dtypes(exclude=[np.number]).columns.tolist()

def encode(X):
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes
    return X.astype(float)

X_test_enc = encode(X_test)

# Small background dataset — summarises the training distribution
background = shap.sample(X_test_enc, 50, random_state=42)

# Rows to explain
X_explain = X_test_enc.sample(100, random_state=42)

# ── Wrap pipeline.predict as a plain function ─────────────────────────────────
def model_predict(X_array):
    """
    Accepts a numpy array, converts to DataFrame with original column names,
    and returns pipeline predictions. KernelExplainer only ever calls this —
    it never inspects pipeline internals.
    """
    X_df = pd.DataFrame(X_array, columns=FEATURE_COLS)
    return pipeline.predict(X_df)

# ── Build KernelExplainer ─────────────────────────────────────────────────────
print("Building KernelExplainer (black-box, sklearn-version-safe)...")
explainer = shap.KernelExplainer(model_predict, background.values)

print("Computing SHAP values for 100 test samples (nsamples=100)...")
shap_values = explainer.shap_values(X_explain.values, nsamples=100)
# shap_values shape: (n_samples, n_features) — a plain numpy array

# ── Plot 1: Beeswarm summary ──────────────────────────────────────────────────
print("Generating beeswarm summary plot...")
plt.figure()
shap.summary_plot(shap_values, X_explain, feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_summary.png")

# ── Plot 2: Bar chart of mean |SHAP| ─────────────────────────────────────────
print("Generating feature importance bar chart...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Mean |SHAP|": mean_abs_shap
}).sort_values("Mean |SHAP|", ascending=True)

fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURE_COLS) * 0.35)))
ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="steelblue")
ax.set_xlabel("Mean absolute SHAP value")
ax.set_title("Global Feature Importance (Random Forest — SHAP)")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_bar.png")

# ── Plot 3: Waterfall for one interesting prediction ─────────────────────────
# Pick the sample whose prediction deviates most from the baseline (most
# informative waterfall — not just a random row).
print("Generating waterfall plot for most-interesting prediction...")
preds      = model_predict(X_explain.values)
base       = explainer.expected_value
deviations = np.abs(preds - base)
idx        = int(np.argmax(deviations))

sv_row  = shap_values[idx]
top_n   = 12
order   = np.argsort(np.abs(sv_row))[::-1][:top_n]
labels  = [FEATURE_COLS[i] for i in order][::-1]
values  = sv_row[order][::-1]
colors  = ["#e74c3c" if v > 0 else "#3498db" for v in values]

fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.4)))
ax.barh(labels, values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (impact on log sales)")
ax.set_title(
    f"Waterfall — sample #{idx}\n"
    f"Baseline: {base:.3f}  |  Prediction: {base + sv_row.sum():.3f}  |  "
    f"Actual: {y_test.iloc[idx]:.3f}"
)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_waterfall.png")

print("\nDone. All SHAP plots saved.")
