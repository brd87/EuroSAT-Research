import torch
import rasterio

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import has_file_allowed_extension

def _default_loader(path):
    with rasterio.open(path) as src:
        image = src.read()

    return torch.from_numpy(image).float()

class EuroSATMS(DatasetFolder):

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=_default_loader,
    ):
        super().__init__(
            root=root,
            loader=loader,
            extensions=(".tif", ".tiff"),
            transform=transform,
            target_transform=target_transform,
        )