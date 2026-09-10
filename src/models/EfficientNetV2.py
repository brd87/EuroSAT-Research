import torch
from torch import nn
from torchvision import models

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.nameid = "EfficientNetV2"

        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_s(weights = weights)
        
        self.backbone.classifier[-1] = nn.Linear(
            self.backbone.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)