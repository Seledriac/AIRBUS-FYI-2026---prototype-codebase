import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import joblib
import json
import os

os.makedirs("models", exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/generated/engine_data.csv")
scaler = joblib.load("models/scaler.pkl")

# True silo: engine 0 only (standard profile, MAX_LIFE=300)
# The model will learn only one degradation pattern and fail to generalise
silo_df = df[df["engine_id"] == 0]
X = silo_df[["cycle", "temperature", "vibration", "pressure"]]
y = silo_df["RUL"]

X_scaled = scaler.transform(X)
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

print("Training LOCAL model (engine 0 only — standard profile silo)...")

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=16,
    verbose=1,
    validation_data=(X_val, y_val),
)

model.save("models/local_model.keras")

with open("models/local_convergence.json", "w") as f:
    json.dump({"val_mae": [float(v) for v in history.history["val_mae"]]}, f)

print("\nLocal model saved → models/local_model.keras")
print("Note: trained on 1 engine (standard), will generalise poorly to other profiles.")
