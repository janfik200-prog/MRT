"""
Почти одинаковые снимки: тот же снимок после пересохранения, отражения, поворота на 180°,
шума или смены яркости, а также соседние срезы той же серии.

    from data_prep.duplicates import thumbnails, similar_pairs, components
    thumbs = thumbnails(prepare(r.path) for r in records)   # [N, 32, 32]
    i, j, sim = similar_pairs(thumbs)                        # пары похожих снимков
    groups = components(len(thumbs), i, j)                   # номер группы для каждого снимка
    copies = components(len(thumbs), i[sim > 0.98], j[sim > 0.98])   # строже — копии одного снимка

Снимки сравниваются по уменьшенным копиям: держать в памяти нужно только их, а не снимки.
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mri.config import NEAR_DUP_SIZE, NEAR_DUP_THRESHOLD

# Варианты, которые считаются тем же снимком. Поворот на 90° не нужен: в данных его нет.
# Работают и для одного снимка [h, w], и для пачки [N, h, w].
ORIENTATIONS = {
    "same": lambda t: t,
    "hflip": lambda t: t[..., :, ::-1],
    "vflip": lambda t: t[..., ::-1, :],
    "rot180": lambda t: t[..., ::-1, ::-1],
}


def thumbnails(images, size: int = NEAR_DUP_SIZE) -> np.ndarray:
    """
    Уменьшенные копии size×size: float32 [N, size, size]. images — любой итератор снимков
    uint8 224×224 (после prepare()), например генератор: снимки не накапливаются в памяти.
    224 / 32 = 7, поэтому INTER_AREA — среднее блоков 7×7: отразить копию — то же самое,
    что уменьшить отражённый снимок.
    """
    return np.stack([cv2.resize(np.ascontiguousarray(x), (size, size), interpolation=cv2.INTER_AREA)
                     for x in images]).astype(np.float32)


def _vectors(thumbs: np.ndarray) -> np.ndarray:
    """[N, size²] с нулевым средним и единичной длиной: скалярное произведение — корреляция,
    она не зависит от яркости и контраста."""
    v = thumbs.reshape(len(thumbs), -1)
    v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


def similar_pairs(thumbs: np.ndarray, threshold: float = NEAR_DUP_THRESHOLD, chunk: int = 2048) -> tuple:
    """
    Пары похожих снимков по уменьшенным копиям (thumbnails): (i, j, sim) — номера снимков и
    корреляция, для всех пар, где снимок i как есть, отражённым или повёрнутым на 180°
    похож на снимок j сильнее threshold. Пара (i, i) не входит.
    """
    base = _vectors(thumbs)
    rows, cols, sims = [], [], []
    for transform in ORIENTATIONS.values():
        variant = _vectors(np.ascontiguousarray(transform(thumbs)))
        for start in range(0, len(thumbs), chunk):  # кусками, чтобы матрица N×N не занимала гигабайты
            sim = variant[start:start + chunk] @ base.T
            i, j = np.nonzero(sim > threshold)
            rows.append(i + start)
            cols.append(j)
            sims.append(sim[i, j])
    i, j, sim = np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)
    other = i != j
    return i[other], j[other], sim[other]


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Номер компоненты для каждого из n снимков, если пары (i, j) связаны: связи замыкаются
    по цепочке (A≈B, B≈C — все трое вместе). Снимок без пар — сам по себе.
    """
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def near_duplicate_groups(thumbs: np.ndarray, threshold: float = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Номер группы почти одинаковых снимков для каждого снимка (корреляция > threshold, по цепочке)."""
    i, j, _ = similar_pairs(thumbs, threshold)
    return components(len(thumbs), i, j)
