# Airbus Federated Learning — Prototype de maintenance prédictive

Prototype de démonstration comparant trois approches d'apprentissage automatique pour la prédiction de la **Remaining Useful Life (RUL)** de moteurs d'avion, avec un focus sur la confidentialité des données.

| Approche | Précision | Confidentialité |
|---|---|---|
| Local (silo) | Mauvaise — 1 moteur seulement | Totale |
| Centralisé | Très bonne — toutes les données centralisées | Nulle |
| **Fédéré** | **Bonne — sans partage de données brutes** | **Totale** |

---

## Prérequis

- Python 3.10 ou 3.11 (TensorFlow ne supporte pas encore Python 3.13)
- pip

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Ordre de lancement

Les scripts sont **dépendants les uns des autres** et doivent être exécutés dans cet ordre précis.

### 1. Générer les données simulées

```bash
python generate_data.py
```

Crée `data/generated/engine_data.csv` — 30 moteurs × 250 cycles avec 3 profils de dégradation (standard, accéléré, normal).

---

### 2. Entraîner le modèle centralisé

```bash
python centralized.py
```

- Entraîne un modèle sur **tous les moteurs** (référence de précision maximale)
- **Génère et sauvegarde le scaler** (`models/scaler.pkl`) — les étapes suivantes en dépendent
- Sauvegarde : `models/centralized_model.keras`, `models/centralized_convergence.json`

---

### 3. Entraîner le modèle local (silo)

```bash
python local.py
```

- Entraîne sur **1 seul moteur** (engine 0, profil standard) — simule un acteur isolé
- Sauvegarde : `models/local_model.keras`, `models/local_convergence.json`

---

### 4. Entraîner le modèle fédéré

```bash
python federated_learning.py
```

- FedAvg : 3 clients × 5 rounds × 5 epochs locales
- Aucune donnée brute ne quitte les clients — seuls les poids agrégés sont partagés
- Sauvegarde : `models/global_model.keras`, `models/federated_convergence.json`

---

### 5. Générer les résultats de comparaison

```bash
python comparison.py
```

- Évalue les 3 modèles sur le même jeu de test
- Sauvegarde : `models/comparison_results.json`

---

### 6. Générer les explications SHAP (optionnel)

```bash
python explainability.py
```

- Produit `models/shap_summary.png` affiché dans le dashboard
- Les features sont labellisées : Cycle moteur, Température (°C), Vibration (g), Pression (bar)

---

### 7. Lancer le dashboard Streamlit

```bash
streamlit run app.py
```

Ouvre automatiquement `http://localhost:8501`

---

## Structure du projet

```
airbus-chat/
│
├── data/
│   └── generated/
│       └── engine_data.csv          # Généré par generate_data.py
│
├── models/                          # Généré automatiquement
│   ├── scaler.pkl
│   ├── local_model.keras
│   ├── centralized_model.keras
│   ├── global_model.keras
│   ├── local_convergence.json
│   ├── centralized_convergence.json
│   ├── federated_convergence.json
│   ├── comparison_results.json
│   └── shap_summary.png
│
├── generate_data.py                 # Simulation de 30 moteurs
├── centralized.py                   # Modèle centralisé + fit scaler
├── local.py                         # Modèle silo (1 moteur)
├── federated_learning.py            # Entraînement fédéré (FedAvg)
├── comparison.py                    # Agrégation des métriques
├── explainability.py                # Graphique SHAP
├── app.py                           # Dashboard Streamlit
└── requirements.txt
```

---

## Dashboard

Le dashboard comporte deux onglets :

** Dashboard**
- Sélection du moteur via la sidebar
- Niveau de risque (LOW / MEDIUM / CRITICAL) basé sur le RUL prédit
- Comparaison côte à côte des 3 graphes RUL prédit vs RUL réel
- Graphique SHAP d'importance des features

**Comparaison IA**
- Tableau MAE / R² / Confidentialité pour les 3 approches
- Graphique de convergence de l'entraînement
