import math
import config

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision (macro)",
    "recall": "Recall (macro)",
    "f1": "F1 (macro)",
    "mcc": "MCC",
    "auc": "AUC (macro, OVR)",
    "avg_loss": "Avg. Loss",
    "baseline": "Baseline Acc.",
}

NARRATIVE_EXCLUDED = ("classification_report")

AVERAGE_RANK_EXCLUDED = ("balanced_accuracy",)

AGGREGATE_KEYS = {"accuracy", "macro avg", "weighted avg"}

BEST_COLOR = "#2e7d32"
WORST_COLOR = "#c62828"
DEFAULT_COLOR = "#5b8ac0"
BEST_CELL = "#c6efce"
WORST_CELL = "#ffc7ce"

FONT_NAME = "Arial"
HEADER_FILL = "1F4E78"
DIAGONAL_FILL = "DDEBF7"

VALUE_FORMAT = "0.0000"

CHECKPOINT_COLUMNS = (
    ("epoch", "Best Epoch", "0"),
    ("train_loss", "Train Loss", VALUE_FORMAT),
    ("valid_loss", "Valid Loss", VALUE_FORMAT),
)

CLASS_REPORT_COLUMNS = (
    ("precision", VALUE_FORMAT),
    ("recall", VALUE_FORMAT),
    ("f1-score", VALUE_FORMAT),
    ("support", "0"),
)

BAR_EXCLUDED = ("baseline",)

def scalar_keys(metrics_merged):
    return list(METRIC_LABELS)
    # return [k for k in config.METRIC_DIRECTIONS.keys() if k not in NARRATIVE_EXCLUDED]

def compared_keys(metrics_merged, metrics_merged_ranked):
    # the metrics that count in rank averages, the verdict and strengths/weaknesses:
    # ranked (None = NaN, e.g. AUC that could not be computed) and not a duplicate of another metric
    return [k for k in scalar_keys(metrics_merged)
            if k not in AVERAGE_RANK_EXCLUDED and metrics_merged_ranked[k] is not None]

def leaderboard(ranks):
    return sorted(range(len(ranks)), key=lambda i: ranks[i])

def best_worst(metrics_merged_ranked, key): # handling ties but might replace with .index() in the return
    ranks = metrics_merged_ranked[key]
    if ranks is None:
        return set(), set()
    best_rank, worst_rank = min(ranks), max(ranks)
    if best_rank == worst_rank:
        return set(), set()
    return ({i for i, r in enumerate(ranks) if r == best_rank}, 
            {i for i, r in enumerate(ranks) if r == worst_rank})

def has_clear_margin(values, idx_a, idx_b, rel_margin=config.DEFAULT_REL_MARGIN, abs_margin=config.DEFAULT_ABS_MARGIN):
    dif = abs(values[idx_a] - values[idx_b])
    margin = max(abs_margin, rel_margin * abs(values[idx_b]))
    return dif >= margin

def extract_class_names(classification_report):
    return [k for k in classification_report.keys() if k not in AGGREGATE_KEYS]


def format_value(value):
    # if value is None or (isinstance(value, float) and math.isnan(value)):
    #     return "N/A"
    # if isinstance(value, float):
    #     return f"{value:.4f}"
    # return str(value)
    return "N/A" if value != value else f"{value:.4f}"
