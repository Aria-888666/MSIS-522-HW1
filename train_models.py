import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from preprocessing import load_data, split_data


def evaluate(y_true, preds):

    rmse = mean_squared_error(y_true, preds, squared=False)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    return rmse, mae, r2


def train_models():

    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = split_data(df)

    if not os.path.exists("models"):
        os.makedirs("models")

    models = {

        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),

        "cart": DecisionTreeRegressor(random_state=42),

        "random_forest": RandomForestRegressor(random_state=42),

        "lightgbm": LGBMRegressor(random_state=42),

        "mlp": MLPRegressor(
            hidden_layer_sizes=(128,128),
            max_iter=300,
            random_state=42
        )
    }

    grids = {

        "cart":{
            "model__max_depth":[3,5,7,10],
            "model__min_samples_leaf":[5,10,20]
        },

        "random_forest":{
            "model__n_estimators":[100,200],
            "model__max_depth":[5,8]
        },

        "lightgbm":{
            "model__n_estimators":[100,200],
            "model__learning_rate":[0.01,0.05,0.1]
        }

    }

    results = []

    for name, model in models.items():

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        if name in grids:

            grid = GridSearchCV(
                pipe,
                grids[name],
                cv=5,
                scoring="neg_mean_squared_error"
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

        else:

            best_model = pipe.fit(X_train, y_train)

        preds = best_model.predict(X_test)

        rmse, mae, r2 = evaluate(y_test, preds)

        results.append([name, rmse, mae, r2])

        joblib.dump(best_model, f"models/{name}.pkl")

    results_df = pd.DataFrame(results, columns=["Model","RMSE","MAE","R2"])

    results_df.to_csv("model_results.csv", index=False)

    print(results_df)


if __name__ == "__main__":
    train_models()
