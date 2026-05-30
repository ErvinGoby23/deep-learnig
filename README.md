# Système d'aide au tri radiologique - Deep Learning

Projet Deep Learning - EFREI Paris - Mastère Data Engineering & IA 2025-2026

## Prérequis

- Python **3.10.11** obligatoire
- 8GB RAM minimum recommandé

## Installation

```bash
# Créer le venv
python -m venv .venv

# Activer le venv
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Lancer la démo Streamlit

```bash
python -m streamlit run demo.py
```

Ouvrir **http://localhost:8501** dans le navigateur, charger une radiographie et voir les prédictions.

## Lancer MLflow

```bash
python mlflow_tracking.py
mlflow ui --backend-store-uri ./mlruns
```

Ouvrir **http://127.0.0.1:5000**, cliquer sur **Model training** puis **Experiments** et sélectionner **chest_xray_classification** pour voir les 7 runs.

## Configuration matérielle

- CPU uniquement (pas de GPU requis)
- Python 3.10.11