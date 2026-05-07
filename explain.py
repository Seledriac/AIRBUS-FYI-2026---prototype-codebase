"""
Explicabilité simple et pédagogique.

On n'utilise pas SHAP pour garder le projet léger.
Méthode utilisée :

1. On repère les points les plus anormaux du vol.
2. On compare leurs features au comportement normal appris.
3. Les features avec les plus grands écarts standardisés expliquent l'anomalie.

C'est une approximation simple, mais compréhensible.
"""

from typing import Dict

import numpy as np
import pandas as pd


def readable_feature_name(feature: str) -> str:
    """Traduit quelques noms techniques en descriptions lisibles."""
    mapping = {
        "vibration_moteur": "niveau instantané de vibration moteur",
        "vibration_delta": "variation brutale de vibration moteur",
        "vibration_rolling_std_5": "instabilité récente de la vibration",
        "vibration_rolling_mean_5": "moyenne récente de vibration",
        "vibration_ecart_moyenne_locale": "écart à la moyenne locale de vibration",
        "altitude": "altitude",
        "altitude_delta": "variation d'altitude",
        "temperature_exterieure": "température extérieure",
        "temperature_delta": "variation de température extérieure",
        "time_minute": "position temporelle dans le vol",
    }

    if feature.startswith("phase_"):
        return f"phase de vol : {feature.replace('phase_', '')}"

    return mapping.get(feature, feature)


def explain_flight(
    flight_id: int,
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    sample_scores: np.ndarray,
    training_stats: Dict,
    top_n_features: int = 5,
) -> Dict:
    """Explique pourquoi un vol est considéré comme anormal."""
    flight_mask = raw_df["flight_id"].values == flight_id

    flight_features = feature_df.loc[flight_mask].copy()
    flight_scores = sample_scores[flight_mask]

    if len(flight_features) == 0:
        raise ValueError(f"flight_id inconnu : {flight_id}")

    # On explique à partir des points les plus anormaux du vol.
    n_top_points = max(1, int(0.15 * len(flight_features)))

    top_indices_local = np.argsort(flight_scores)[-n_top_points:]
    suspicious_points = flight_features.iloc[top_indices_local]

    means = pd.Series(training_stats["feature_means"])
    stds = pd.Series(training_stats["feature_stds"]).replace(0, 1e-6)

    zscores = ((suspicious_points - means).abs() / stds)
    contribution = zscores.mean().sort_values(ascending=False)

    top_features = []

    for feature, value in contribution.head(top_n_features).items():
        top_features.append(
            {
                "feature": feature,
                "description": readable_feature_name(feature),
                "contribution_score": float(value),
                "normal_mean": float(means[feature]),
                "flight_suspicious_mean": float(suspicious_points[feature].mean()),
            }
        )

    most_anomalous_time_index = int(np.argmax(flight_scores))
    flight_rows = raw_df.loc[flight_mask].reset_index(drop=True)
    most_anomalous_row = flight_rows.iloc[most_anomalous_time_index].to_dict()

    return {
        "flight_id": int(flight_id),
        "max_sample_score": float(np.max(flight_scores)),
        "mean_top_sample_score": float(
            np.mean(np.sort(flight_scores)[-n_top_points:])
        ),
        "most_anomalous_time_minute": int(most_anomalous_row["time_minute"]),
        "most_anomalous_phase": most_anomalous_row["phase_vol"],
        "top_contributing_features": top_features,
    }


def explanation_to_text(explanation: Dict) -> str:
    """Convertit une explication en texte lisible."""
    lines = []

    lines.append(f"Vol {explanation['flight_id']}")
    lines.append(f"- minute la plus suspecte : {explanation['most_anomalous_time_minute']}")
    lines.append(f"- phase correspondante    : {explanation['most_anomalous_phase']}")
    lines.append(f"- score max point         : {explanation['max_sample_score']:.4f}")
    lines.append("- principales causes possibles :")

    for item in explanation["top_contributing_features"]:
        lines.append(
            f"  * {item['description']} "
            f"(score contribution {item['contribution_score']:.2f}, "
            f"moyenne normale {item['normal_mean']:.2f}, "
            f"moyenne points suspects {item['flight_suspicious_mean']:.2f})"
        )

    return "\n".join(lines)