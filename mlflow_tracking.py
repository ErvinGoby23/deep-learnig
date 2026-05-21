import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('mlflow_figures', exist_ok=True)

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("chest_xray_classification")

experiments = [
    {
        "name": "CNN_Scratch",
        "auc": 0.6350, "f1": 0.0986, "recall": 0.9643, "precision": 0.0540,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0172, 0.0141, 0.0137, 0.0134, 0.0133, 0.0130],
        "val_losses":   [0.0127, 0.0124, 0.0124, 0.0127, 0.0121, 0.0122],
        "model_path": "models/cnn_scratch.pth"
    },
    {
        "name": "ResNet18",
        "auc": 0.5990, "f1": 0.0978, "recall": 0.9277, "precision": 0.0537,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0148, 0.0135, 0.0133, 0.0131, 0.0133, 0.0131],
        "val_losses":   [0.0381, 0.0188, 0.0131, 0.0122, 0.0140, 0.0199],
        "model_path": "models/cnn_transfer.pth"
    },
    {
        "name": "ViT",
        "auc": 0.5849, "f1": 0.0977, "recall": 0.9286, "precision": 0.0536,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0134, 0.0130, 0.0128, 0.0128, 0.0128, 0.0127],
        "val_losses":   [0.0129, 0.0126, 0.0126, 0.0128, 0.0125, 0.0125],
        "model_path": "models/vit.pth"
    },
    {
        "name": "Image_seule",
        "auc": 0.6379, "f1": 0.0990, "recall": 0.9448, "precision": 0.0543,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0137, 0.0127, 0.0126, 0.0126, 0.0125, 0.0125],
        "val_losses":   [0.0120, 0.0118, 0.0120, 0.0119, 0.0117, 0.0118],
        "model_path": "models/model_image.pth"
    },
    {
        "name": "Texte_seul",
        "auc": 0.4885, "f1": 0.0973, "recall": 0.9405, "precision": 0.0534,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0158, 0.0131, 0.0128, 0.0127, 0.0126, 0.0125],
        "val_losses":   [0.0127, 0.0120, 0.0118, 0.0117, 0.0116, 0.0115],
        "model_path": "models/model_text.pth"
    },
    {
        "name": "Multimodal",
        "auc": 0.6239, "f1": 0.0972, "recall": 0.9085, "precision": 0.0533,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "FocalLoss",
        "train_losses": [0.0137, 0.0127, 0.0126, 0.0126, 0.0125, 0.0125],
        "val_losses":   [0.0120, 0.0118, 0.0120, 0.0119, 0.0117, 0.0118],
        "model_path": "models/model_multimodal.pth"
    },
    {
        "name": "Autoencoder",
        "auc": None, "f1": None, "recall": None, "precision": None,
        "epochs": 6, "lr": 1e-3, "batch_size": 32, "loss": "MSELoss",
        "train_losses": [0.0321, 0.0075, 0.0059, 0.0052, 0.0047, 0.0044],
        "val_losses":   [0.0321, 0.0075, 0.0059, 0.0052, 0.0047, 0.0044],
        "model_path": "models/autoencoder.pth"
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_param("model_name", exp["name"])
        mlflow.log_param("epochs", exp["epochs"])
        mlflow.log_param("learning_rate", exp["lr"])
        mlflow.log_param("batch_size", exp["batch_size"])
        mlflow.log_param("image_size", 64)
        mlflow.log_param("train_size", 10000)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", exp["loss"])
        mlflow.log_param("threshold", 0.1)
        mlflow.log_param("augmentation", "HorizontalFlip+Rotation+ColorJitter")
        mlflow.log_param("sampling", "stratified_10k")

        if exp["auc"] is not None:
            mlflow.log_metric("auc_mean", exp["auc"])
            mlflow.log_metric("f1_macro", exp["f1"])
            mlflow.log_metric("recall_macro", exp["recall"])
            mlflow.log_metric("precision_macro", exp["precision"])
        else:
            mlflow.log_metric("reconstruction_error_mean", 0.0042)
            mlflow.log_metric("anomaly_threshold", 0.0081)
            mlflow.log_metric("anomalies_detected", 91)

        for i, (tl, vl) in enumerate(zip(exp["train_losses"], exp["val_losses"])):
            mlflow.log_metric("train_loss", tl, step=i)
            mlflow.log_metric("val_loss", vl, step=i)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(exp["train_losses"], label="Train Loss")
        ax.plot(exp["val_losses"], label="Val Loss")
        ax.set_title(f"{exp['name']} - Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        fig_path = f"mlflow_figures/loss_{exp['name']}.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)
        mlflow.log_artifact(exp["model_path"])

        print(f"Run loggé : {exp['name']}")

print("\nTous les runs MLflow enregistrés ")