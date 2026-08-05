from torchvision.datasets import EuroSAT
from torchvision import transforms


def get_eurosat_rgb(root="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = EuroSAT(
        root=root,
        download=True,
        transform=transform,
    )

    return dataset