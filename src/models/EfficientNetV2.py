import torch
from torch import nn
from torchvision import models
import config

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=config.CLASSES, in_channels=3, optimizer=None):
        super().__init__()
        self.nameid = "EfficientNetV2"

        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_s(weights = weights)

        if in_channels != 3:
            self.backbone.features[0][0] = self._adapt_first_conv(self.backbone.features[0][0], in_channels)

        self.backbone.classifier[-1] = nn.Linear(
            self.backbone.classifier[-1].in_features,
            num_classes
        )

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LR)

    def forward(self, x):
        return self.backbone(x)

    def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, rgb_idx=(3, 2, 1)) -> nn.Conv2d:
        new = nn.Conv2d(
            in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=conv.bias is not None,
        )
        with torch.no_grad():
            w = conv.weight                                   # [out, 3, k, k]
            new_w = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            for i, band in enumerate(rgb_idx):                # B04,B03,B02 -> R,G,B
                new_w[:, band] = w[:, i]
            new.weight.copy_(new_w * 3 / in_channels)         # keep activation scale similar
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new