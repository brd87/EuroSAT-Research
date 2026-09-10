import os
from pathlib import Path

from torchvision.datasets import EuroSAT
from torchvision import transforms
from custom_data.multispectral.dataset import EuroSATMS
from custom_data.multispectral.download import download


def get_eurosat_rgb(root:Path = None):
    if root is None:
            root = Path(__file__).resolve().parents[2] / "data/dsready/rbg"
    root.mkdir(exist_ok = True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = EuroSAT(
        root=root,
        download=True,
        transform=transform,
    )

    return dataset

def get_eurosat_ms(root:Path = None):
    if root is None:
            root = Path(__file__).resolve().parents[2] / "data/dsready/ms"
    root.mkdir(exist_ok = True)

    download(root)

    dataset = EuroSATMS(root=root)
    return dataset

