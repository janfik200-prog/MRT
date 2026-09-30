"""
Аугментации для датасета Brain Tumor MRI.

Работает "на лету": исходные файлы в dataset/ не изменяются и не копируются.

Уровни (буквы комбинируются):
    none   - только предобработка (для валидации/теста и для бейзлайна)
    A      - геометрия + яркость
    AB     - A + шум и качество (против shortcut в notumor)
    ABC    - AB + МРТ-специфичные артефакты (bias field, ghosting, motion)
    ABCG   - ABC + доп. геометрия (мягкий crop, эластика/сеточные искажения)
    ABCGI  - ABCG + доп. интенсивность (CLAHE, резкость) — "максимальная" ручная политика
    R      - RandAugment-подобная: flip + 2 случайные операции из общего пула

Использование:
    from augment import load_image, build_transforms
    tf = build_transforms("AB")
    img = load_image(path)            # grayscale uint8, мозг обрезан по контуру
    out = tf(image=img)["image"]      # 224x224 uint8
"""
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # без проверки обновлений в сети
import albumentations as A
import cv2
import numpy as np

IMG_SIZE = 224
LEVELS = ("none", "A", "AB", "ABC", "ABCG", "ABCGI", "R")  # основные варианты для экспериментов


# ----------------------------------------------------------------------------
# 0. Предобработка (одинаково для train и test)
# ----------------------------------------------------------------------------
def crop_brain(img: np.ndarray, thresh: int = 20, margin: int = 5) -> np.ndarray:
    """Обрезает чёрные поля: bbox наибольшего светлого контура + отступ."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < 0.05 * img.shape[0] * img.shape[1]:  # контур подозрительно мал — не режем
        return img
    H, W = img.shape[:2]
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, W), min(y + h + margin, H)
    return img[y0:y1, x0:x1]


def load_image(path, crop: bool = True) -> np.ndarray:
    """Читает любой файл (RGB/L/RGBA/P) как grayscale uint8 и обрезает поля."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return crop_brain(img) if crop else img


def _resize() -> list:
    """Вписать в квадрат IMG_SIZE с сохранением пропорций, добить чёрным."""
    return [
        A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ]


# ----------------------------------------------------------------------------
# C. МРТ-специфичные артефакты (свои реализации, без torch/TorchIO)
# ----------------------------------------------------------------------------
class RandomBiasField(A.ImageOnlyTransform):
    """Неоднородность поля: плавное мультипликативное изменение яркости."""

    def __init__(self, coef: float = 0.3, p: float = 0.5):
        super().__init__(p=p)
        self.coef = coef

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        c = self.random_generator.uniform(-self.coef, self.coef, size=6)
        log_field = c[0] * xx + c[1] * yy + c[2] * xx * yy + c[3] * xx**2 + c[4] * yy**2 + c[5]
        return {"field": np.exp(log_field).astype(np.float32)}

    def apply(self, img, field=None, **params):
        return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("coef",)


class RandomGhosting(A.ImageOnlyTransform):
    """Ghosting: слабые смещённые копии изображения вдоль оси фазового кодирования."""

    def __init__(self, intensity=(0.05, 0.2), num_ghosts=(2, 4), p: float = 0.5):
        super().__init__(p=p)
        self.intensity, self.num_ghosts = intensity, num_ghosts

    def get_params_dependent_on_data(self, params, data):
        g = self.random_generator
        return {
            "k": int(g.integers(self.num_ghosts[0], self.num_ghosts[1] + 1)),
            "a": float(g.uniform(*self.intensity)),
            "axis": int(g.integers(0, 2)),
        }

    def apply(self, img, k=2, a=0.1, axis=0, **params):
        f = img.astype(np.float32)
        n = img.shape[axis]
        out = f.copy()
        for i in range(1, k + 1):
            out += a / i * np.roll(f, i * n // (k + 1), axis=axis)
        out *= f.max() / max(out.max(), 1e-6)
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("intensity", "num_ghosts")


class RandomMotion(A.ImageOnlyTransform):
    """Артефакт движения: смесь нескольких слегка сдвинутых/повёрнутых копий."""

    def __init__(self, max_shift: float = 4.0, max_angle: float = 3.0, n: int = 3, p: float = 0.5):
        super().__init__(p=p)
        self.max_shift, self.max_angle, self.n = max_shift, max_angle, n

    def get_params_dependent_on_data(self, params, data):
        g = self.random_generator
        moves = [
            (g.uniform(-self.max_shift, self.max_shift),
             g.uniform(-self.max_shift, self.max_shift),
             g.uniform(-self.max_angle, self.max_angle))
            for _ in range(self.n)
        ]
        return {"moves": moves}

    def apply(self, img, moves=(), **params):
        h, w = img.shape[:2]
        acc = img.astype(np.float32)
        for dx, dy, ang in moves:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            M[:, 2] += (dx, dy)
            acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        return np.clip(acc / (len(moves) + 1), 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("max_shift", "max_angle", "n")


# ----------------------------------------------------------------------------
# Наборы аугментаций по уровням
# ----------------------------------------------------------------------------
def aug_A() -> list:
    """Геометрия + яркость."""
    return [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-15, 15),
            translate_percent=(-0.1, 0.1),
            scale=(0.9, 1.1),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.8,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]


def aug_B() -> list:
    """Шум и качество: разрушают подсказки по формату/разрешению файла."""
    return [
        A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
        A.ImageCompression(quality_range=(50, 95), p=0.5),
        A.Downscale(scale_range=(0.5, 0.9), p=0.3),
    ]


def aug_C() -> list:
    """МРТ-специфичные артефакты."""
    return [
        RandomBiasField(coef=0.3, p=0.3),
        A.OneOf([RandomGhosting(p=1.0), RandomMotion(p=1.0)], p=0.2),
    ]


def aug_G() -> list:
    """Доп. геометрия: мягкий random-resized-crop и нелинейные деформации."""
    bc = dict(border_mode=cv2.BORDER_CONSTANT, fill=0)
    return [
        # scale >= 0.75: маленькая опухоль гипофиза не вырезается
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=6, **bc, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.15, **bc, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, **bc, p=1.0),
        ], p=0.3),
    ]


def aug_I() -> list:
    """Доп. интенсивность: локальный контраст и резкость (разные протоколы/сканеры)."""
    return [
        A.OneOf([
            A.CLAHE(clip_limit=(1, 3), p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), p=1.0),
            A.UnsharpMask(alpha=(0.2, 0.5), p=1.0),
        ], p=0.3),
    ]


def aug_R() -> list:
    """RandAugment-подобная политика: flip + 2 случайные операции из всего пула (A, G, I, B, C)."""
    bc = dict(border_mode=cv2.BORDER_CONSTANT, fill=0)
    pool = [
        A.Affine(rotate=(-15, 15), **bc, p=1.0),
        A.Affine(translate_percent=(-0.1, 0.1), **bc, p=1.0),
        A.Affine(scale=(0.85, 1.15), **bc, p=1.0),
        A.Affine(shear=(-8, 8), **bc, p=1.0),
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.ElasticTransform(alpha=30, sigma=6, **bc, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        A.CLAHE(clip_limit=(1, 3), p=1.0),
        A.Sharpen(alpha=(0.1, 0.3), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
        A.ImageCompression(quality_range=(40, 95), p=1.0),
        A.Downscale(scale_range=(0.4, 0.9), p=1.0),
        RandomBiasField(coef=0.3, p=1.0),
        RandomGhosting(p=1.0),
        RandomMotion(p=1.0),
    ]
    return [A.HorizontalFlip(p=0.5), A.SomeOf(pool, n=2, replace=False, p=1.0)]


def build_transforms(level: str = "A", normalize: bool = False) -> A.Compose:
    """
    level: "none" | "R" | комбинация букв из "ABCGI" в любом порядке, например "A", "AB", "ABC", "ABCGI".
        A — геометрия + яркость, B — шум/качество файла, C — МРТ-артефакты,
        G — доп. геометрия (crop, эластика), I — доп. интенсивность (CLAHE, резкость),
        R — RandAugment-подобная политика (исключает остальные буквы).
    Вход — результат load_image(). Выход — IMG_SIZE x IMG_SIZE, 1 канал.
    normalize=True — float32 в диапазоне ~[-1, 1] (для обучения).
    Для предобученных CNN (3 канала) повторите канал: np.repeat(x[..., None], 3, -1).
    """
    if level != "none" and level != "R" and (not level or set(level) - set("ABCGI")):
        raise ValueError(f"level: 'none', 'R' или буквы из 'ABCGI', получено {level!r}")
    # Сначала вписываем в квадрат: аугментации работают на 224x224 (быстрее)
    # и равномерно затрагивают и снимок, и поля (нет "рамки" после яркости).
    t = _resize()
    if level == "R":
        t += aug_R()
    # Порядок как в реальности: геометрия/анатомия → интенсивность → артефакты
    # съёмки (C) → сохранение файла (B: шум, JPEG, даунскейл).
    if "A" in level:
        t += aug_A()
    if "G" in level:
        t += aug_G()
    if "I" in level:
        t += aug_I()
    if "C" in level:
        t += aug_C()
    if "B" in level:
        t += aug_B()
    if normalize:
        t.append(A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0))
    return A.Compose(t)


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
