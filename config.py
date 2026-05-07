from pathlib import Path

# =========================
# Dossiers du projet
# =========================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# =========================
# Fichiers
# =========================

DATA_FILE = DATA_DIR / "flights_synthetic.csv"

MODEL_FILE = MODEL_DIR / "isolation_forest_model.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"
METADATA_FILE = MODEL_DIR / "training_metadata.joblib"

# =========================
# Simulation des vols
# =========================

RANDOM_SEED = 42

N_FLIGHTS = 50
FLIGHT_DURATION_MINUTES = 100
ANOMALOUS_FLIGHT_RATIO = 0.30

# Phases de vol utilisées dans le dataset
PHASES = [
    "decollage",
    "montee",
    "croisiere",
    "descente",
    "atterrissage",
]

# =========================
# Modèle Isolation Forest
# =========================

N_ESTIMATORS = 200

# Le modèle est entraîné uniquement sur vols normaux.
# contamination indique au modèle qu'une petite fraction des points normaux
# peut tout de même paraître atypique.
CONTAMINATION = 0.03

# Seuil final au niveau "vol".
# On calibre le seuil avec les vols normaux uniquement.
THRESHOLD_PERCENTILE_ON_NORMAL_FLIGHTS = 95

# Pour obtenir un score par vol, on prend la moyenne des points les plus suspects.
TOP_SCORE_FRACTION = 0.15