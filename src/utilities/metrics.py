from dataclasses import dataclass

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

@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    fmt: str = "0.000"
    abs_threshold: float = 0.02 # deliberately configurable tresholds (not for statistical significance tests)
    relative_threshold: float = 0.02
    rankable: bool = True

METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("avg_loss", "Average loss", False, "0.0000", 0.02, 0.02),
    MetricSpec("accuracy", "Accuracy", True, "0.0000", 0.02, 0.02),
    MetricSpec("balanced_accuracy", "Balanced accuracy", True, "0.0000", 0.02, 0.02),
    MetricSpec("precision", "Macro precision", True, "0.0000", 0.02, 0.02),
    MetricSpec("recall", "Macro recall", True, "0.0000", 0.02, 0.02),
    MetricSpec("f1", "Macro F1", True, "0.0000", 0.02, 0.02),
    MetricSpec("mcc", "MCC", True, "0.0000", 0.05, 0.02),
    MetricSpec("auc", "Macro AUC", True, "0.0000", 0.02, 0.02),
)


def calculate(result, class_names, nameid):
    logits = result["logits"]
    targets = result["targets"]

    labels = list(range(len(class_names)))

    preds = torch.argmax(logits, dim=1).numpy()
    
    metrics = {
        "model_id": nameid,
        "avg_loss": result["avg_loss"],
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average='macro', zero_division=0),
        "recall": recall_score(targets, preds, average='macro', zero_division=0),
        "f1": f1_score(targets, preds, average='macro', zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "baseline": np.bincount(targets).max() / len(targets),
        "predictions": preds
    }
    
    try:
        probs = torch.softmax(logits, dim=1).numpy()
        metrics["probabilities"] = probs
        metrics['auc'] = roc_auc_score(targets, probs, average='macro', multi_class='ovr')
    except ValueError:
        metrics["probabilities"] = None
        metrics['auc'] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(targets, preds, labels=labels)
    metrics["classification_report"] = classification_report(
        result["targets"].numpy().ravel(),
        preds,
        labels=labels,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return metrics


def print_all(metrics, checkpoint):
    print("\n--- FINAL METRICS ---")
    print(f"MODEL: {metrics['model_id']}")
    print(f"BEST CHECKPOINT: Epoch {checkpoint["epoch"]} | avg_train_loss {checkpoint["train_loss"]} | avg_valid_loss {checkpoint["valid_loss"]}")

    print(f"Average loss:        {metrics['avg_loss']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"Macro precision:     {metrics['precision']:.4f}")
    print(f"Macro recall:        {metrics['recall']:.4f}")
    print(f"Macro F1:            {metrics['f1']:.4f}")
    print(f"MCC:                 {metrics['mcc']:.4f}")
    print(f"Baseline accuracy:   {metrics['baseline']:.4f}")
    print(f"AUC:                 {metrics['auc']:.4f}")

    print("\nClassification report:")
    print(metrics["classification_report"])


def merge(metrics_set) -> dict[str, list[any]]:# what a lovely usage example of comprehensions and unpacking
    metrics_set = list(metrics_set)
    keys = set().union(*(metrics.keys() for metrics in metrics_set))

    metrics_merged = {}
    for key in keys:
        metrics_merged[key] = [
            metrics[key]
            for metrics in metrics_set
            if key in metrics
        ]

    return metrics_merged


def merged_rank(metrics_merged, reverse=True):
    metrics_merged_ranked = {}
    for key, values in metrics_merged.items():
        if any(isinstance(val, str) for val in values):
            metrics_merged_ranked[key] = None

        elif all(isinstance(val, dict) for val in values):
            metrics_merged_ranked[key] = __recursive_rank(values, reverse)

        elif not any(isinstance(val, (list, np.ndarray, dict)) for val in values):
            metrics_merged_ranked[key] = sorted(range(len(values)), key = lambda i: values[i], reverse=reverse)
        
        else:
            metrics_merged_ranked[key] = None

    return metrics_merged_ranked


def __recursive_rank(values, reverse=True):
    if all(isinstance(v, dict) for v in values): # check for inner classes
        keys = set().union(*(v.keys() for v in values))
        return {
            key: __recursive_rank(
                [v[key] for v in values if key in v],
                reverse=reverse
            )
            for key in keys
        }
    
    return sorted(range(len(values)), key = lambda i: values[i], reverse=reverse)