from pathlib import Path

import torch
import rasterio

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import has_file_allowed_extension

def __default_loader(path):
    with rasterio.open(path) as src:
        image = src.read()

    return torch.from_numpy(image).float()

class EuroSATMS(VisionDataset):
    classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake"
    ]

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    extensions = (".tif", ".tiff")

    def __init__(self, root, transform=None, target_transform=None, transforms=None, loader=__default_loader):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)

        root = Path(root)
        self.loader = loader

        self.samples = []
        self.targets = []

        for class_name in self.classes:
            class_dir = root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"ERROR - missing directory: {class_dir}")

            for file in sorted(class_dir.iterdir()):
                if has_file_allowed_extension(
                    file.name,
                    self.extensions,
                ):
                    target = self.class_to_idx[class_name]

                    self.samples.append((file, target))
                    self.targets.append(target)

        if len(self.samples) == 0:
            raise RuntimeError("ERROR - found 0 images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
            path, target = self.samples[index]

            image = self.loader(path)

            if self.transforms is not None:
                image, target = self.transforms(image,target)

            else:
                if self.transform is not None:
                    image = self.transform(image)

                if self.target_transform is not None:
                    target = self.target_transform(target)

            return image, target

