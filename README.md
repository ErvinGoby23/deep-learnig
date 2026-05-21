# Système d'aide au tri radiologique - Deep Learning

Projet Deep Learning - EFREI Paris - Mastère Data Engineering & IA 2025-2026

## Prérequis

- Python **3.10.x** obligatoire
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

Ouvrir **http://127.0.0.1:5000**

## Ordre d'exécution des notebooks

1. `eda_chestmnist.ipynb` — Analyse exploratoire
2. `preprocessing.ipynb` — Preprocessing et augmentation
3. `cnn_scratch.ipynb` — CNN from scratch
4. `cnn_transfer.ipynb` — ResNet18 Transfer Learning
5. `vit.ipynb` — Vision Transformer
6. `autoencoder.ipynb` — Détection d'anomalies
7. `multimodal.ipynb` — Multimodalité image + texte

## Configuration matérielle

- CPU uniquement (pas de GPU requis)
- Python 3.10.11
- Temps d'entraînement : ~10-15 min par modèle (6 epochs, 10k images)

## Choix techniques

- **Loss** : FocalLoss (gamma=2, alpha=0.25)
- **Seuil décision** : 0.1 (optimisé pour le recall en médical)
- **Augmentation** : HorizontalFlip, Rotation(10°), ColorJitter
- **Sampling** : Stratified 10 000 images
- **Seed** : 42
- **Early stopping** : patience=3
