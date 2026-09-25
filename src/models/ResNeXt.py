import torch
from torch import nn
from torchvision import models
import config

class ResNeXt(nn.Module):
    def __init__(self, num_classes=config.CLASSES):
        super().__init__()
        self.nameid = "ResNeXt"

        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        self.backbone = models.resnext50_32x4d(weights=weights)

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)