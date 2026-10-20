"""
Предобработка снимков (одинаковая для train и test) — это ещё не аугментация.

    from mri.preprocess import prepare
    img = prepare(path)   # uint8 224x224, grayscale

Здесь используется OpenCV: чтение файлов, поиск контура головы и уменьшение
размера — это работа с файлами и форматами, а не формулы аугментации.
"""
import cv2
import numpy as np

from .config import IMG_SIZE


def crop_brain(img: np.ndarray, thresh: int = 20, margin: int = 5) -> np.ndarray:
    """Обрезает чёрные поля: прямоугольник вокруг самого большого светлого пятна + отступ."""
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


def fit_square(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Вписать в квадрат size×size с сохранением пропорций и добить чёрным по краям."""
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (size - nh) // 2, (size - nw) // 2
    return np.pad(img, ((top, size - nh - top), (left, size - nw - left)))  # pad нулями = чёрным


def prepare(path) -> np.ndarray:
    """Файл -> uint8 224x224: grayscale, обрезка полей, вписывание в квадрат."""
    return fit_square(load_image(path))
