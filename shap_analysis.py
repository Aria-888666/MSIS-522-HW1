import shap
import joblib

from preprocessing import load_data, split_data

df = load_data()

X_train, X_test, y_train, y_test, preprocessor = split_data(df)

model = joblib.load("models/random_forest.pkl")

X_sample = X_test.sample(200, random_state=42)

X_processed = model.named_steps["prep"].transform(X_sample)

explainer = shap.Explainer(model.named_steps["model"])

shap_values = explainer(X_processed)

shap.summary_plot(shap_values)

shap.plots.bar(shap_values)

shap.plots.waterfall(shap_values[0])
