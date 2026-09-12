import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
import torch

import config

def classification_metrics(result):
    logits = result["logits"]
    targets = result["targets"]

    preds = torch.argmax(logits, dim=1).numpy()

    metrics = {
        "avg_loss": result["avg_loss"],
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average='macro', zero_division=0),
        "recall": recall_score(targets, preds, average='macro', zero_division=0),
        "f1": f1_score(targets, preds, average='macro', zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "baseline": np.bincount(targets).max() / len(targets),
    }

    try:
        probs = torch.softmax(logits, dim=1).numpy()
        metrics['auc'] = roc_auc_score(targets, probs, average='macro', multi_class='ovr')
    except ValueError:
        metrics['auc'] = np.nan

    metrics["predictions"] = preds
    metrics["probabilities"] = logits

    return metrics

def print_metrics(result, metrics, checkpoint):


    print("\n--- FINAL METRICS ---")
    print(f"BEST CHECKPOINT: Epoch {checkpoint["epoch"]} | avg_train_loss {checkpoint["train_loss"]} | avg_valid_loss {checkpoint["valid_loss"]}")
    print(
        classification_report(
            result["targets"].numpy().ravel(),
            metrics["predictions"],
            digits=4,
        )
    )