"""
Поиск снимков в dataset/.

    from data_prep.scan import list_images
    records = list_images("Training")      # все источники, все классы

Датасет один, но лежит по папкам источников (см. SOURCES в mri/config.py):
    dataset/<источник>/<Training|Testing>/<класс>/<файл>
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from mri.config import CLASSES, DATASET_DIR, IMAGE_EXTENSIONS, ROOT, SOURCES


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
    Порядок: источники как в sources (по умолчанию SOURCES), внутри — классы как в CLASSES,
    внутри — файлы по имени.
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


def sha256(path) -> str:
    """Отпечаток содержимого файла: одинаковые файлы — одинаковая строка."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
