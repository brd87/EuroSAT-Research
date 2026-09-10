from pathlib import Path 
import shutil
from urllib.request import urlretrieve 
from zipfile import ZipFile
import config


def download(root: Path = None):
    if not _simple_check(root=root): 
        zip_path = root.parent / "EuroSAT_MS.zip"

        print("Downloading EuroSAT MS...") 
        urlretrieve(config.EUROSAT_MS_URL, zip_path) 

        print("Extracting EuroSAT MS...") 
        with ZipFile(zip_path, "r") as zip_file: 
            zip_file.extractall(root.parent) 

        extracted_path = root.parent / "EuroSAT_MS"
        if not extracted_path.exists():
            raise RuntimeError(f"ERROR - expected extracted directory not found: {extracted_path}")

        shutil.move( str(extracted_path), str(root))

        zip_path.unlink()

def _simple_check(root: Path = None) -> bool:
    class_dirs = [
        path for path in root.iterdir()
        if path.is_dir()
    ]

    return len(class_dirs) == 10
    