"""
Поиск снимков в dataset/ и загрузка в память.

    from mri.data import list_images, load_images
    records = list_images("Training")      # все источники, все классы
    x, y = load_images(records)            # uint8 [N, 224, 224], метки [N]

Датасет один, но лежит по папкам источников (см. SOURCES в config.py):
    dataset/<источник>/<Training|Testing>/<класс>/<файл>
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASSES, DATASET_DIR, IMAGE_EXTENSIONS, ROOT, SOURCES
from .preprocess import prepare


@dataclass(frozen=True)
class ImageRecord:
    """Один снимок: путь, источник, Training/Testing и номер класса в CLASSES."""
    path: Path
    source: str
    split: str
    label: int

    @property
    def cls(self) -> str:
        return CLASSES[self.label]

    @property
    def file(self) -> str:
        """Путь относительно корня проекта, с прямыми слешами: dataset/PMRAM/Training/..."""
        return self.path.relative_to(ROOT).as_posix()


def list_images(split: str, sources=None) -> list:
    """
    Все снимки split ("Training" или "Testing") по источникам и классам.
    Порядок: источники как в SOURCES, внутри — классы как в CLASSES, внутри — файлы по имени.
    """
    records = []
    for source in sources or SOURCES:
        for label, cls in enumerate(CLASSES):
            folder = DATASET_DIR / source / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"Нет папки: {folder}")
            records += [ImageRecord(f, source, split, label) for f in sorted(folder.iterdir())
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    return records


def load_images(records) -> tuple:
    """Снимки после prepare() и их метки: (uint8 [N, 224, 224], int [N])."""
    x = np.stack([prepare(r.path) for r in records])
    y = np.array([r.label for r in records])
    return x, y


def sha256(path) -> str:
    """Отпечаток содержимого файла: одинаковые файлы — одинаковая строка."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
