import shap
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("models/global_model.keras")
scaler = joblib.load("models/scaler.pkl")


FEATURE_NAMES = ["Cycle moteur", "Température (°C)", "Vibration (g)", "Pression (bar)"]

df = pd.read_csv("data/generated/engine_data.csv")

X = df[["cycle", "temperature", "vibration", "pressure"]]

X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_NAMES)


explainer = shap.Explainer(model, X_scaled.iloc[:100])

shap_values = explainer(X_scaled.iloc[:50])


plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("models/shap_summary.png", bbox_inches='tight')

print("SHAP explanation generated.")