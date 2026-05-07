"""
Fonctions utilitaires :
- création des dossiers
- sauvegarde / chargement
- calcul des scores d'anomalie
- agrégation des scores au niveau vol
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config


def ensure_directories() -> None:
    """Crée les dossiers nécessaires."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_joblib(obj, path: Path) -> None:
    """Sauvegarde un objet Python avec joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path):
    """Charge un objet sauvegardé avec joblib."""
    return joblib.load(path)


def anomaly_scores_from_model(model, X: np.ndarray) -> np.ndarray:
    """
    Convertit le score Isolation Forest en score d'anomalie.

    Dans scikit-learn :
    - decision_function élevé = plus normal
    - decision_function faible = plus anormal

    On inverse donc le signe pour obtenir :
    - score élevé = plus anormal
    """
    return -model.decision_function(X)


def aggregate_scores_by_flight(
    df: pd.DataFrame,
    sample_scores: np.ndarray,
    sample_predictions: np.ndarray,
    top_fraction: float = config.TOP_SCORE_FRACTION,
) -> pd.DataFrame:
    """
    Agrège les scores point par point en un score par vol.

    Pour éviter qu'un seul point isolé domine, on prend la moyenne des
    points les plus suspects du vol.
    """
    temp = df[["flight_id", "vol_defaillant"]].copy()
    temp["sample_anomaly_score"] = sample_scores
    temp["sample_predicted_anomaly"] = sample_predictions.astype(int)

    rows = []

    for flight_id, group in temp.groupby("flight_id"):
        scores = group["sample_anomaly_score"].values

        n_top = max(1, int(len(scores) * top_fraction))
        top_scores = np.sort(scores)[-n_top:]

        rows.append(
            {
                "flight_id": flight_id,
                "true_failure": int(group["vol_defaillant"].max()),
                "flight_anomaly_score": float(np.mean(top_scores)),
                "max_sample_anomaly_score": float(np.max(scores)),
                "sample_anomaly_ratio": float(
                    group["sample_predicted_anomaly"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def calibrate_flight_threshold(
    normal_flight_scores: pd.Series,
    percentile: float,
) -> float:
    """Calibre un seuil de score de vol à partir des vols normaux."""
    return float(np.percentile(normal_flight_scores.values, percentile))


def save_text(path: Path, content: str) -> None:
    """Sauvegarde du texte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")