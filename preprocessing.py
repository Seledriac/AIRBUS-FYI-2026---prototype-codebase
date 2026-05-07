"""
Préparation des données pour Isolation Forest.

Le modèle ne travaille pas uniquement sur les colonnes brutes.
On ajoute des variables simples qui décrivent le comportement temporel :

- variation de vibration
- moyenne glissante de vibration
- écart-type glissant de vibration
- variation d'altitude
- variation de température

Ces variables permettent de détecter un changement de pattern.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class Preprocessor:
    """Objet contenant le scaler et la liste des features."""
    scaler: StandardScaler
    feature_columns: List[str]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features temporelles calculées par vol."""
    df = df.copy()
    df = df.sort_values(["flight_id", "time_minute"]).reset_index(drop=True)

    grouped = df.groupby("flight_id", group_keys=False)

    df["altitude_delta"] = grouped["altitude"].diff().fillna(0)
    df["temperature_delta"] = grouped["temperature_exterieure"].diff().fillna(0)
    df["vibration_delta"] = grouped["vibration_moteur"].diff().fillna(0)

    df["vibration_rolling_mean_5"] = grouped["vibration_moteur"].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )

    df["vibration_rolling_std_5"] = grouped["vibration_moteur"].transform(
        lambda s: s.rolling(window=5, min_periods=1).std().fillna(0)
    )

    df["vibration_ecart_moyenne_locale"] = (
        df["vibration_moteur"] - df["vibration_rolling_mean_5"]
    )

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la table de features numériques utilisée par le modèle."""
    df = add_temporal_features(df)

    feature_df = pd.DataFrame(index=df.index)

    numeric_columns = [
        "time_minute",
        "altitude",
        "temperature_exterieure",
        "vibration_moteur",
        "altitude_delta",
        "temperature_delta",
        "vibration_delta",
        "vibration_rolling_mean_5",
        "vibration_rolling_std_5",
        "vibration_ecart_moyenne_locale",
    ]

    for col in numeric_columns:
        feature_df[col] = df[col].astype(float)

    # Encodage simple des phases de vol.
    for phase in config.PHASES:
        feature_df[f"phase_{phase}"] = (df["phase_vol"] == phase).astype(int)

    return feature_df


def fit_preprocessor(
    df_normal: pd.DataFrame,
) -> Tuple[Preprocessor, pd.DataFrame, np.ndarray]:
    """
    Apprend le scaler uniquement sur des données normales.

    Returns:
        preprocessor : objet contenant scaler + colonnes
        feature_df   : features non standardisées
        X            : features standardisées
    """
    feature_df = build_feature_table(df_normal)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)

    preprocessor = Preprocessor(
        scaler=scaler,
        feature_columns=list(feature_df.columns),
    )

    return preprocessor, feature_df, X


def transform_with_preprocessor(
    df: pd.DataFrame,
    preprocessor: Preprocessor,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Transforme un dataset avec le scaler appris."""
    feature_df = build_feature_table(df)

    # Garantit le même ordre de colonnes qu'à l'entraînement.
    feature_df = feature_df[preprocessor.feature_columns]

    X = preprocessor.scaler.transform(feature_df.values)

    return feature_df, X