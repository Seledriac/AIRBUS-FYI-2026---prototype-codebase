import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

os.makedirs("models", exist_ok=True)


# =========================
# LOAD & SCALE (master scaler)
# =========================

df = pd.read_csv("data/generated/engine_data.csv")
X = df[["cycle", "temperature", "vibration", "pressure"]]
y = df["RUL"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# =========================
# MODEL
# =========================

model = Sequential([
    Input(shape=(4,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# =========================
# TRAIN
# =========================

print("Training CENTRALIZED model (all 30 engines)...")

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    verbose=1,
    validation_data=(X_val, y_val),
)

model.save("models/centralized_model.keras")

with open("models/centralized_convergence.json", "w") as f:
    json.dump({"val_mae": [float(v) for v in history.history["val_mae"]]}, f)

print("\nCentralized model saved → models/centralized_model.keras")
print("Scaler saved            → models/scaler.pkl")
