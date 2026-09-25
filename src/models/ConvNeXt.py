import torch
from torch import nn
from torchvision import models
import config

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=config.CLASSES):
        super().__init__()
        self.nameid = "ConvNeXt"

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights = weights)
        
        self.backbone.classifier[-1] = nn.Linear(
            self.backbone.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)