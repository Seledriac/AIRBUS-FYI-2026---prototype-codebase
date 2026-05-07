"""
Entraînement du modèle non supervisé Isolation Forest.

Important :
Le modèle est entraîné uniquement sur les vols normaux.
Les vols défaillants ne sont jamais utilisés pour entraîner le modèle.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from preprocessing import fit_preprocessor, transform_with_preprocessor
from utils import (
    aggregate_scores_by_flight,
    anomaly_scores_from_model,
    calibrate_flight_threshold,
    ensure_directories,
    save_joblib,
)


def main() -> None:
    ensure_directories()

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {config.DATA_FILE}. "
            "Lancez d'abord generate_data.py."
        )

    df = pd.read_csv(config.DATA_FILE)
    df = df.sort_values(["flight_id", "time_minute"]).reset_index(drop=True)

    # Entraînement uniquement sur les vols normaux.
    normal_df = df[df["vol_defaillant"] == 0].copy()

    if normal_df.empty:
        raise ValueError("Aucun vol normal disponible pour l'entraînement.")

    preprocessor, normal_feature_df, X_train = fit_preprocessor(normal_df)

    model = IsolationForest(
        n_estimators=config.N_ESTIMATORS,
        contamination=config.CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )

    model.fit(X_train)

    # Calibration du seuil de score de vol sur les vols normaux.
    _, X_normal = transform_with_preprocessor(normal_df, preprocessor)

    sample_scores = anomaly_scores_from_model(model, X_normal)
    sample_predictions = model.predict(X_normal) == -1

    normal_flight_scores = aggregate_scores_by_flight(
        normal_df,
        sample_scores,
        sample_predictions,
    )

    threshold = calibrate_flight_threshold(
        normal_flight_scores["flight_anomaly_score"],
        config.THRESHOLD_PERCENTILE_ON_NORMAL_FLIGHTS,
    )

    # Statistiques utilisées pour expliquer les anomalies.
    training_stats = {
        "feature_means": normal_feature_df.mean().to_dict(),
        "feature_stds": normal_feature_df.std().replace(0, 1e-6).to_dict(),
        "flight_threshold": threshold,
    }

    save_joblib(model, config.MODEL_FILE)
    save_joblib(preprocessor, config.PREPROCESSOR_FILE)
    save_joblib(training_stats, config.METADATA_FILE)

    print("Modèle entraîné et sauvegardé.")
    print(f"Modèle       : {config.MODEL_FILE}")
    print(f"Preprocessor : {config.PREPROCESSOR_FILE}")
    print(f"Seuil vol    : {threshold:.4f}")


if __name__ == "__main__":
    main()