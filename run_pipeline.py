"""
Exécute tout le prototype de bout en bout :

1. Génération des données
2. Entraînement du modèle Isolation Forest
3. Évaluation du modèle
"""

import subprocess
import sys


def run(command):
    """Exécute une commande Python et arrête le pipeline en cas d'erreur."""
    print("\n" + "=" * 80)
    print("Commande :", " ".join(command))
    print("=" * 80)

    subprocess.check_call(command)


def main() -> None:
    python = sys.executable
    run(['rm', '-rf', 'data', 'models', 'results'])
    run([python, "generate_data.py"])
    run([python, "train_model.py"])
    run([python, "evaluate_model.py"])


if __name__ == "__main__":
    main()