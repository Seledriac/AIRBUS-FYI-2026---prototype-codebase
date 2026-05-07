"""
Évaluation du modèle sur tous les vols.

Ce script :
- charge le modèle entraîné
- calcule un score d'anomalie pour chaque point temporel
- agrège les scores par vol
- calcule accuracy, precision, recall, F1-score
- génère une matrice de confusion
- affiche quelques exemples de vols détectés comme défaillants
- explique les détections
- sauvegarde les résultats
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from explain import explain_flight, explanation_to_text
from preprocessing import transform_with_preprocessor
from utils import (
    aggregate_scores_by_flight,
    anomaly_scores_from_model,
    ensure_directories,
    load_joblib,
    save_text,
)


def save_confusion_matrix_plot(cm, path) -> None:
    """Sauvegarde une matrice de confusion sous forme d'image."""
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm)

    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Défaillant"])
    ax.set_yticklabels(["Normal", "Défaillant"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ensure_directories()

    df = pd.read_csv(config.DATA_FILE)
    df = df.sort_values(["flight_id", "time_minute"]).reset_index(drop=True)

    model = load_joblib(config.MODEL_FILE)
    preprocessor = load_joblib(config.PREPROCESSOR_FILE)
    metadata = load_joblib(config.METADATA_FILE)

    feature_df, X = transform_with_preprocessor(df, preprocessor)

    sample_scores = anomaly_scores_from_model(model, X)
    sample_predictions = model.predict(X) == -1

    flight_results = aggregate_scores_by_flight(
        df,
        sample_scores,
        sample_predictions,
    )

    threshold = metadata["flight_threshold"]

    flight_results["predicted_failure"] = (
        flight_results["flight_anomaly_score"] > threshold
    ).astype(int)

    y_true = flight_results["true_failure"].values
    y_pred = flight_results["predicted_failure"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics_text = (
        "Performances au niveau vol\n"
        "==========================\n"
        f"Accuracy  : {accuracy:.3f}\n"
        f"Precision : {precision:.3f}\n"
        f"Recall    : {recall:.3f}\n"
        f"F1-score  : {f1:.3f}\n"
        f"Seuil score de vol : {threshold:.4f}\n"
        "\nMatrice de confusion [lignes=reel, colonnes=predit]\n"
        f"{cm}\n"
    )

    print(metrics_text)

    # Sauvegarde des résultats globaux
    flight_results.to_csv(
        config.RESULTS_DIR / "flight_results.csv",
        index=False,
    )

    # Sauvegarde des scores point par point
    sample_output = df[["flight_id", "time_minute", "phase_vol"]].copy()
    sample_output["sample_anomaly_score"] = sample_scores
    sample_output["sample_predicted_anomaly"] = sample_predictions.astype(int)

    sample_output.to_csv(
        config.RESULTS_DIR / "sample_scores.csv",
        index=False,
    )

    save_text(config.RESULTS_DIR / "metrics.txt", metrics_text)

    save_confusion_matrix_plot(
        cm,
        config.RESULTS_DIR / "confusion_matrix.png",
    )

    # Exemples de vols détectés comme défaillants
    detected = flight_results[flight_results["predicted_failure"] == 1].copy()
    detected = detected.sort_values(
        "flight_anomaly_score",
        ascending=False,
    ).head(5)

    explanations = []
    explanation_texts = []

    print("Exemples de vols détectés comme défaillants")
    print("==========================================")

    for _, row in detected.iterrows():
        exp = explain_flight(
            flight_id=int(row["flight_id"]),
            raw_df=df,
            feature_df=feature_df,
            sample_scores=sample_scores,
            training_stats=metadata,
        )

        explanations.append(exp)

        text = explanation_to_text(exp)
        explanation_texts.append(text)

        print(text)
        print()

    with open(
        config.RESULTS_DIR / "explanations.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    save_text(
        config.RESULTS_DIR / "explanations.txt",
        "\n\n".join(explanation_texts) if explanation_texts else "Aucun vol détecté.",
    )

    print(f"Résultats sauvegardés dans : {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
    