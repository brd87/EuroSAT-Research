import torch
from torch import nn

import config
import utilities.pipeline as pipeline
import utilities.dataset_get as dataset_get
import utilities.metrics as metrics
import utilities.report as report

from models.ConvNeXt import ConvNeXt
from models.EfficientNetV2 import EfficientNetV2
from models.ResNeXt import ResNeXt


def main():
    print("START")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------- DATA -----------------
    dataset = dataset_get.eurosat_rgb()

    # ----------------- MODEL -----------------
    models = [
        ConvNeXt(num_classes=config.CLASSES),
        EfficientNetV2(num_classes=config.CLASSES),
        ResNeXt(num_classes=config.CLASSES)
        ]

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=config.LR) 
        for model in models
        ]

    criterion = nn.CrossEntropyLoss()

    metrics_set = {}
    for model, optimizer in zip(models, optimizers):
        metrics_result = pipeline.ran(model, dataset, device, criterion, optimizer)
        metrics_set.add(metrics_result)

    metrics_merged = metrics.merge(metrics_set)
    metrics_merged_ranked = metrics.merged_rank(metrics_merged, reverse=True)



if __name__ == "__main__":
    main()

