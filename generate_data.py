import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_ENGINES = 30
N_CYCLES = 250

# Each profile has a different maximum life → local silo model won't generalize
MAX_LIFE = {"standard": 300, "accelerated": 150, "normal": 250}

os.makedirs("data/generated", exist_ok=True)


def simulate_engine(engine_id, profile):
    data = []

    base_temp = 600
    base_vibration = 20
    base_pressure = 100

    for cycle in range(N_CYCLES):
        if profile == "standard":
            degradation = cycle * 0.15

        elif profile == "accelerated":
            degradation = cycle * 0.30

        else:
            degradation = cycle * 0.20

        temp = base_temp + degradation + np.random.normal(0, 3)
        vibration = base_vibration + degradation * 0.4 + np.random.normal(0, 1)
        pressure = base_pressure - degradation * 0.2 + np.random.normal(0, 1)

        rul = max(MAX_LIFE[profile] - cycle + np.random.normal(0, 5), 0)

        data.append([
            engine_id,
            cycle,
            temp,
            vibration,
            pressure,
            rul,
            profile
        ])

    columns = [
        "engine_id",
        "cycle",
        "temperature",
        "vibration",
        "pressure",
        "RUL",
        "profile"
    ]

    return pd.DataFrame(data, columns=columns)


all_data = []

profiles = ["standard", "accelerated", "normal"]

for i in range(N_ENGINES):
    profile = profiles[i % len(profiles)]
    df = simulate_engine(i, profile)
    all_data.append(df)

final_df = pd.concat(all_data)

final_df.to_csv("data/generated/engine_data.csv", index=False)

print("Synthetic dataset generated successfully.")