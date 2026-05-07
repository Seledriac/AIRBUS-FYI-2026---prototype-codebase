"""
Génération de données synthétiques de vols.

Objectif :
- Créer plusieurs vols simulés.
- Certains vols sont normaux.
- Certains vols contiennent une défaillance moteur.
- La défaillance modifie le pattern de vibration moteur.
- La détection ne dépend pas seulement d'un seuil fixe de vibration.

Sortie :
    data/flights_synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd

import config


def get_phase(time_minute: int, duration: int) -> str:
    """Retourne la phase de vol en fonction du temps."""
    ratio = time_minute / max(duration - 1, 1)

    if ratio < 0.10:
        return "decollage"
    if ratio < 0.25:
        return "montee"
    if ratio < 0.70:
        return "croisiere"
    if ratio < 0.90:
        return "descente"
    return "atterrissage"


def altitude_for_phase(
    time_minute: int,
    duration: int,
    phase: str,
    rng: np.random.Generator,
) -> float:
    """Simule une altitude cohérente avec la phase de vol."""
    ratio = time_minute / max(duration - 1, 1)
    cruise_altitude = 11000.0 + rng.normal(0, 200)

    if phase == "decollage":
        local_ratio = ratio / 0.10
        altitude = 1000.0 * local_ratio

    elif phase == "montee":
        local_ratio = (ratio - 0.10) / 0.15
        altitude = 1000.0 + (cruise_altitude - 1000.0) * local_ratio

    elif phase == "croisiere":
        altitude = cruise_altitude + rng.normal(0, 80)

    elif phase == "descente":
        local_ratio = (ratio - 0.70) / 0.20
        altitude = cruise_altitude * (1.0 - local_ratio)

    else:
        local_ratio = (ratio - 0.90) / 0.10
        altitude = 1200.0 * (1.0 - local_ratio)

    return max(0.0, altitude + rng.normal(0, 30))


def normal_vibration_for_phase(phase: str) -> float:
    """Niveau de vibration typique selon la phase de vol."""
    base = {
        "decollage": 2.8,
        "montee": 2.2,
        "croisiere": 1.35,
        "descente": 1.7,
        "atterrissage": 2.5,
    }
    return base[phase]


def simulate_flight(
    flight_id: int,
    duration: int,
    is_failure: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simule un vol complet."""
    rows = []

    # Température au sol différente pour chaque vol
    ground_temperature = rng.normal(18, 8)

    # La défaillance commence généralement en croisière ou en descente.
    failure_start = None
    failure_strength = 0.0

    if is_failure:
        failure_start = int(
            rng.integers(
                int(duration * 0.38),
                int(duration * 0.75),
            )
        )
        failure_strength = rng.uniform(0.55, 1.10)

    previous_vibration = None

    for t in range(duration):
        phase = get_phase(t, duration)
        altitude = altitude_for_phase(t, duration, phase, rng)

        # La température diminue avec l'altitude.
        temperature = ground_temperature - 0.0065 * altitude + rng.normal(0, 1.5)

        # Vibration normale dépendante de la phase.
        vibration = normal_vibration_for_phase(phase)
        vibration += rng.normal(0, 0.12)

        # Petite influence environnementale.
        vibration += 0.004 * max(0, 15 - temperature)

        in_failure_period = False

        if is_failure and failure_start is not None and t >= failure_start:
            in_failure_period = True
            dt = t - failure_start

            # Défaillance = changement de pattern :
            # oscillations, instabilité, variation brutale.
            # Ce n'est pas seulement une vibration trop élevée.
            oscillation = (
                failure_strength * np.sin(dt * 0.85)
                + 0.35 * np.sin(dt * 2.1)
            )

            instability = rng.normal(0, 0.25 + 0.012 * dt)

            pattern_shift = (
                failure_strength
                * 0.35
                * np.sign(np.sin(dt * 0.25))
            )

            vibration = vibration + (oscillation + instability + pattern_shift)

        # Lissage léger pour garder un signal réaliste.
        if previous_vibration is not None:
            vibration = 0.65 * vibration + 0.35 * previous_vibration

        previous_vibration = vibration

        rows.append(
            {
                "flight_id": flight_id,
                "time_minute": t,
                "phase_vol": phase,
                "altitude": altitude,
                "temperature_exterieure": temperature,
                "vibration_moteur": vibration,

                # Labels uniquement pour évaluer le prototype.
                # Le modèle non supervisé ne les utilise pas à l'entraînement.
                "vol_defaillant": int(is_failure),
                "point_apres_debut_defaillance": int(in_failure_period),
            }
        )

    return pd.DataFrame(rows)


def generate_dataset(
    n_flights: int,
    duration: int,
    anomalous_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Génère plusieurs vols synthétiques."""
    rng = np.random.default_rng()

    n_anomalous = int(round(n_flights * anomalous_ratio))
    anomalous_ids = set(
        rng.choice(
            np.arange(n_flights),
            size=n_anomalous,
            replace=False,
        )
    )


    flights = []

    for flight_id in range(n_flights):
        is_failure = flight_id in anomalous_ids
        flight_df = simulate_flight(
            flight_id=flight_id,
            duration=duration,
            is_failure=is_failure,
            rng=rng,
        )
        flights.append(flight_df)

    return pd.concat(flights, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generer un dataset synthetique de vols."
    )
    parser.add_argument("--n-flights", type=int, default=config.N_FLIGHTS)
    parser.add_argument("--duration", type=int, default=config.FLIGHT_DURATION_MINUTES)
    parser.add_argument(
        "--anomalous-ratio",
        type=float,
        default=config.ANOMALOUS_FLIGHT_RATIO,
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)

    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        n_flights=args.n_flights,
        duration=args.duration,
        anomalous_ratio=args.anomalous_ratio,
        seed=args.seed,
    )

    df.to_csv(config.DATA_FILE, index=False)

    print(f"Dataset genere : {config.DATA_FILE}")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de vols : {df['flight_id'].nunique()}")
    print(f"Vols defaillants : {df.groupby('flight_id')['vol_defaillant'].max().sum()}")


if __name__ == "__main__":
    main()