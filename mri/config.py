"""Общие настройки проекта: пути, классы, источники снимков. Меняются только здесь."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"               # исходные снимки (не изменяются кодом)

# Всё, что создаётся при работе (результаты, временные файлы), —
# только в _work/. Папка не попадает в git; её можно удалить целиком.
WORK_DIR = ROOT / "_work"
RESULTS_DIR = WORK_DIR / "results"               # обучение: results/<дата_время>/ (model.ipynb)
SCRATCH_DIR = WORK_DIR / "scratch"               # разовые проверки и черновики

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SPLITS = ("Training", "Testing")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
IMG_SIZE = 224

# Датасет один, но собран из нескольких источников — по папке на каждый:
#     dataset/<источник>/<Training|Testing>/<класс>/*.jpg
# Значение — уровень аугментации для Training этого источника (см. mri/augment.py).
# Порядок важен: в нём снимки идут в обучение и в таблицы результатов.
SOURCES = {
    "BDNeuro": "none",
    "PMRAM": "R",
    "MultiClass": "R",
    "KaggleBrainTumor": "R",
}

TARGET = 0.85  # цель: precision и recall не ниже этого значения по каждому классу
