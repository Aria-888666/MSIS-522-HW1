"""
preprocessing.py
----------------
Loads and splits the OJ dataset, and builds the sklearn preprocessing pipeline.

⚠️  SKLEARN VERSION WARNING
The preprocessor returned by split_data() is a fitted-able ColumnTransformer
containing SimpleImputer objects. When this pipeline is saved via joblib.dump()
inside train_models.py, it is pickled with the sklearn version active at
training time. If the sklearn version at runtime (e.g. on Streamlit Cloud)
differs from the training version, loading the pickle will raise:

    AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'

FIX: Pin scikit-learn to the SAME version in requirements.txt as the one
you train with locally. Check your version by running:

    python -c "import sklearn; print(sklearn.__version__)"

Then set that exact version in requirements.txt:

    scikit-learn==X.X.X

After updating requirements.txt, retrain all models:

    python train_models.py

Then commit BOTH the new .pkl files AND requirements.txt to GitHub.
The Streamlit app (Streamlit_app.py) avoids this issue entirely by training
fresh models from the CSV at runtime — but train_models.py and shap_analysis.py
still rely on these pickles, so the version must match.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

TARGET = "logmove"


def load_data(path: str = "oj.csv") -> pd.DataFrame:
    """
    Load the OJ dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to "oj.csv" in the working directory.

    Returns
    -------
    pd.DataFrame
        The raw dataset with all original columns intact.
    """
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def split_data(df: pd.DataFrame):
    """
    Split the dataset and build the preprocessing pipeline.

    Preprocessing steps:
    - Numeric columns: median imputation → standard scaling
    - Categorical columns: most-frequent imputation → one-hot encoding

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including the target column.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices for training (70%) and test (30%) sets.
    y_train, y_test : pd.Series
        Target vectors for training and test sets.
    preprocessor : ColumnTransformer
        Unfitted preprocessing pipeline. Fit on X_train inside the model
        Pipeline to avoid data leakage.

    Notes
    -----
    The preprocessor is returned UNFITTED so it can be embedded inside a
    sklearn Pipeline:

        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)   # preprocessor fits on X_train only

    This ensures the scaler and imputer statistics are computed only from
    training data, preventing data leakage from the test set.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Detect column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Numeric pipeline: impute missing values with median, then standardise
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categorical pipeline: impute with most frequent value, then one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",   # drop any columns not explicitly listed above
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
    )

    print(f"[preprocessing] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"[preprocessing] Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"[preprocessing] Categorical cols ({len(categorical_cols)}): {categorical_cols}")

    return X_train, X_test, y_train, y_test, preprocessor
