import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os

# =========================
# SHARED TEST SET
# =========================

df = pd.read_csv("data/generated/engine_data.csv")
scaler = joblib.load("models/scaler.pkl")

X = df[["cycle", "temperature", "vibration", "pressure"]]
y = df["RUL"]
X_scaled = scaler.transform(X)

_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# =========================
# LOAD MODELS
# =========================

for path in ["models/local_model.keras", "models/centralized_model.keras", "models/global_model.keras"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run the corresponding training script first.")

local_model    = tf.keras.models.load_model("models/local_model.keras")
central_model  = tf.keras.models.load_model("models/centralized_model.keras")
fed_model      = tf.keras.models.load_model("models/global_model.keras")


# =========================
# EVALUATE (same test set)
# =========================

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test, verbose=0)
    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 1),
        "r2":  round(float(r2_score(y_test, pred)), 3),
    }

metrics_local   = evaluate(local_model,   X_test, y_test)
metrics_central = evaluate(central_model, X_test, y_test)
metrics_fed     = evaluate(fed_model,     X_test, y_test)


# =========================
# LOAD CONVERGENCE DATA
# =========================

def load_json(path, key):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)[key]

conv_local      = load_json("models/local_convergence.json",       "val_mae")
conv_central    = load_json("models/centralized_convergence.json", "val_mae")
round_metrics   = load_json("models/federated_convergence.json",   "round_metrics")


# =========================
# COMPILE & SAVE
# =========================

results = {
    "metrics": {
        "local":       {**metrics_local,   "privacy": True},
        "centralized": {**metrics_central, "privacy": False},
        "federated":   {**metrics_fed,     "privacy": True},
    },
    "convergence": {
        "local":       conv_local,
        "centralized": conv_central,
    },
    "round_metrics": round_metrics,
}

with open("models/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Résultats de comparaison ===")
print(f"Local       MAE: {metrics_local['mae']:>7}   R²: {metrics_local['r2']:>6}   ✅ Privé")
print(f"Centralisé  MAE: {metrics_central['mae']:>7}   R²: {metrics_central['r2']:>6}   ❌ Non-privé")
print(f"Fédéré      MAE: {metrics_fed['mae']:>7}   R²: {metrics_fed['r2']:>6}   ✅ Privé")
print("\nSauvegardé → models/comparison_results.json")
