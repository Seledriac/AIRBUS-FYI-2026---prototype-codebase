import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os

os.makedirs("models", exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/generated/engine_data.csv")
X_all = df[["cycle", "temperature", "vibration", "pressure"]]
y_all = df["RUL"]

scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X_all)

# Hold-out sets shared with other models for fair comparison
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)

NUM_CLIENTS = 3
client_data = np.array_split(np.arange(len(X_scaled)), NUM_CLIENTS)

GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 5


# =========================
# HELPERS
# =========================

def create_model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def average_weights(weight_list):
    avg_weights = []
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(np.array(weights), axis=0))
    return avg_weights


# =========================
# FEDERATED TRAINING (FedAvg)
# =========================

print(f"Federated training — {NUM_CLIENTS} clients, {GLOBAL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs\n")

global_model = create_model()
round_metrics = []

for round_num in range(GLOBAL_ROUNDS):
    print(f"Global Round {round_num + 1}/{GLOBAL_ROUNDS}")
    local_weights = []

    for client_id, indices in enumerate(client_data):
        X_client = X_scaled[indices]
        y_client = y_all.iloc[indices]
        X_tr, _, y_tr, _ = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42
        )
        client_model = create_model()
        client_model.set_weights(global_model.get_weights())
        client_model.fit(X_tr, y_tr, epochs=LOCAL_EPOCHS, batch_size=32, verbose=0)
        local_weights.append(client_model.get_weights())
        print(f"  Client {client_id} trained")

    global_model.set_weights(average_weights(local_weights))

    pred_train = global_model.predict(X_train, verbose=0)
    pred_test  = global_model.predict(X_test,  verbose=0)

    mae_train = round(float(mean_absolute_error(y_train, pred_train)), 2)
    mae_test  = round(float(mean_absolute_error(y_test,  pred_test)),  2)
    r2_test   = round(float(r2_score(y_test, pred_test)), 4)

    round_metrics.append({
        "round":     round_num + 1,
        "mae_train": mae_train,
        "mae_test":  mae_test,
        "r2_test":   r2_test,
    })
    print(f"  → MAE Train: {mae_train:.2f}  MAE Test: {mae_test:.2f}  R² Test: {r2_test:.4f}\n")


# =========================
# SAVE
# =========================

global_model.save("models/global_model.keras")

with open("models/federated_convergence.json", "w") as f:
    json.dump({"round_metrics": round_metrics}, f)

print("Federated model saved → models/global_model.keras")
