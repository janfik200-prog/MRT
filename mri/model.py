"""Свёрточная сеть SimpleCNN (~394 тыс. параметров)."""
import torch.nn as nn

from .config import CLASSES


def block(c_in: int, c_out: int) -> nn.Sequential:
    """Свёртка 3×3 → нормализация → ReLU → уменьшение в 2 раза."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


def simple_cnn(n_classes: int = len(CLASSES), dropout: float = 0.3) -> nn.Sequential:
    """
    5 блоков (224 → 7, каналы 16 → 256), затем среднее по каждому каналу,
    Dropout и линейный слой на n_classes. Вход: [B, 1, 224, 224].
    """
    return nn.Sequential(
        block(1, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes),
    )
