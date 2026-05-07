# Prototype IA de détection de défaillance moteur d'avion

## Objectif

Ce projet est un prototype simple de maintenance prédictive aéronautique.

On simule plusieurs vols d'avion. Chaque vol contient une série temporelle de mesures :

- phase de vol
- altitude
- température extérieure
- vibration moteur

Certains vols sont normaux. D'autres contiennent une défaillance moteur qui modifie brutalement le **pattern de vibration**.

Le système ne cherche donc pas seulement une vibration trop élevée. Il cherche un comportement anormal par rapport aux vols normaux.

Le modèle utilisé est **Isolation Forest**, un modèle non supervisé simple.

Le modèle est entraîné uniquement avec des vols normaux, puis il est testé sur des vols normaux et défaillants.

---

## Structure des fichiers

```text
.
├── config.py              # paramètres du projet
├── generate_data.py       # génération des vols synthétiques
├── preprocessing.py       # création des features et standardisation
├── train_model.py         # entraînement Isolation Forest sur vols normaux
├── evaluate_model.py      # évaluation, métriques, matrice de confusion
├── explain.py             # explication simple des anomalies
├── utils.py               # fonctions utilitaires
├── run_pipeline.py        # exécution complète du pipeline
├── requirements.txt       # dépendances Python
└── README.md              # documentation