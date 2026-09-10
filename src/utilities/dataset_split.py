import os
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
import joblib

import config

def __make_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, eval_loader, test_loader

def __split(dataset):
    indices = list(range(len(dataset)))
    targets = dataset.targets

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        stratify=targets,
        random_state=config.SEED,
    )

    temp_targets = [targets[i] for i in temp_indices]

    test_ratio = (
        config.TEST_RATIO /
        (config.VAL_RATIO + config.TEST_RATIO)
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio,
        stratify=temp_targets,
        random_state=config.SEED,
    )

    return train_indices, val_indices, test_indices

def subset(dataset:Dataset, scaler_path):
    dataset.x = dataset.x.copy()
    train_indices, val_indices, test_indices = __split(dataset)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader, val_loader, test_loader = __make_loaders(train_dataset, val_dataset, test_dataset)

    return train_loader, val_loader, test_loader