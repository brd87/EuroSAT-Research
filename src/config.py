EXPERIMENT_NAME = "0.4"
CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
TRAIN_RATIO = 0.7
TEST_RATIO = 0.1
VAL_RATIO = 0.2
CLASSIFICATION_THRESHOLD = 0.5
SEED = 2137
EUROSAT_MS_URL = ('https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1')

CLOSE_CALL_THRESHOLD = 0.5
DEFAULT_REL_MARGIN = 0.03
DEFAULT_ABS_MARGIN = 1e-6

METRIC_DIRECTIONS = {
    "avg_loss": False,
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "mcc": True,
    "auc": True,
    
    "classification_report": True,  # merged_rank() needs a direction for every
                                     # dict-valued metric; applies to nested
                                     # precision/recall/f1-score/support leaves.
}