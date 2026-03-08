"""
train_models.py
---------------
Trains all models and saves them as sklearn Pipelines.

IMPORTANT — sklearn version consistency:
  Models are saved via joblib.dump as full Pipeline objects. When loaded later
  (in the Streamlit app or shap_analysis.py), the sklearn version at runtime
  MUST match the version used here. A mismatch causes errors like:
      AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'

  To prevent this, requirements.txt pins the sklearn version. If you retrain
  locally, make sure your environment matches requirements.txt exactly:
      pip install -r requirements.txt
  Then retrain:
      python train_models.py
  Then commit the new .pkl files to GitHub.
"""

import os
import numpy as np
import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from preprocessing import load_data, split_data

print(f"sklearn version: {sklearn.__version__}")
print("Make sure this matches the version in requirements.txt and on Streamlit Cloud.\n")


def evaluate(y_true, preds):
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae  = mean_absolute_error(y_true, preds)
    r2   = r2_score(y_true, preds)
    return rmse, mae, r2


def train_models():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = split_data(df)

    os.makedirs("models", exist_ok=True)

    # ── Model definitions ────────────────────────────────────────────────────
    models = {
        "linear":        LinearRegression(),
        "ridge":         Ridge(),
        "lasso":         Lasso(),
        "cart":          DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42),
        "lightgbm":      LGBMRegressor(random_state=42, verbose=-1),
        "mlp":           MLPRegressor(
                             hidden_layer_sizes=(128, 128),
                             max_iter=300,
                             random_state=42
                         ),
    }

    # ── Hyperparameter grids ─────────────────────────────────────────────────
    grids = {
        "cart": {
            "model__max_depth":        [3, 5, 7, 10],
            "model__min_samples_leaf": [5, 10, 20],
        },
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [5, 8],
        },
        "lightgbm": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },
    }

    results = []

    for name, model in models.items():
        print(f"Training: {name} ...", end=" ", flush=True)

        pipe = Pipeline([
            ("prep",  preprocessor),
            ("model", model),
        ])

        if name in grids:
            search = GridSearchCV(
                pipe,
                grids[name],
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"best params: {search.best_params_}")
        else:
            best_model = pipe.fit(X_train, y_train)
            print("done")

        preds = best_model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, preds)
        results.append({"Model": name, "RMSE": round(rmse, 4),
                         "MAE": round(mae, 4), "R2": round(r2, 4)})

        # Save the full pipeline — preprocessor + model together
        save_path = f"models/{name}.pkl"
        joblib.dump(best_model, save_path)
        print(f"  → Saved to {save_path}  |  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # ── Save results table ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)
    print("\n── Model Comparison ─────────────────────────────────────────────")
    print(results_df.to_string(index=False))
    print("\nDone. All models saved to models/ and results to model_results.csv")
    print(f"\nRemember: these .pkl files require sklearn=={sklearn.__version__}")
    print("Add this exact version to requirements.txt before pushing to GitHub.")


if __name__ == "__main__":
    train_models()
