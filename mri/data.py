"""
Чтение манифеста и загрузка снимков в память.

    from mri.data import read_manifest, load_images
    manifest = read_manifest()                         # по строке на снимок (см. data_prep/split.py)
    train = manifest[manifest["split"] == "train"]
    x = load_images(train["file"])                     # uint8 [N, 224, 224]

Манифест создаёт data_prep/dataset.ipynb: какие снимки идут в train/test и с какими пометками.
"""
import numpy as np
import pandas as pd

from .config import MANIFEST, ROOT
from .preprocess import prepare


def read_manifest(path=MANIFEST) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Нет манифеста {path}: сначала выполните data_prep/dataset.ipynb")
    return pd.read_csv(path)


def load_images(files) -> np.ndarray:
    """Снимки после prepare(): uint8 [N, 224, 224]; files — пути относительно корня проекта."""
    return np.stack([prepare(ROOT / f) for f in files])
