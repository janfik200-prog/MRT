"""
Балансировка классов при обучении: WeightedRandomSampler.

Проблема. В train классы представлены неравно: glioma примерно в 2.3 раза чаще, чем notumor.
Если брать снимки просто по очереди (shuffle), сеть чаще видит glioma и привыкает отвечать
«glioma» в сомнительных случаях — recall редких классов падает.

Решение. DataLoader берёт снимки не по очереди, а случайно, с весами: у снимка редкого класса
вес больше, поэтому его выбирают чаще. Вес снимка = 1 / (сколько примеров его класса за эпоху).
Тогда сумма весов каждого класса одинакова, и за эпоху классы встречаются примерно поровну.

    from mri.balance import balanced_sampler
    sampler = balanced_sampler(train_data, seed=0)
    train_dl = DataLoader(train_data, batch_size=64, sampler=sampler)   # shuffle не указывать

Что важно знать:
- Выбор «с возвращением»: за эпоху один снимок может попасться несколько раз, а другой — ни разу.
  Редкие классы повторяются чаще, частые — реже.
- Повторы не одинаковые: аугментация (mri/augment.py) срабатывает при каждом взятии снимка,
  поэтому каждый повтор — новый случайный вариант.
- Разных снимков от этого не прибавляется: снимков notumor по-прежнему столько же, их просто
  показывают чаще. Поэтому на них выше риск переобучения.
- Размер эпохи не меняется (по умолчанию len(dataset)), меняется только доля классов.
- Test не балансируется: метрики считаются на настоящем распределении.
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def sample_labels(dataset) -> np.ndarray:
    """
    Класс каждого примера dataset в том порядке, в каком DataLoader видит индексы 0..len-1.
    dataset — MRIDataset или ConcatDataset из них. У MRIDataset с expand=3 снимки идут три раза
    подряд по кругу: индекс i — это снимок i % len(x), поэтому метки повторяются целиком (np.tile).
    """
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    return np.concatenate([np.tile(np.asarray(part.y), part.expand) for part in parts])


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Вес каждого примера: 1 / число примеров его класса. Сумма весов у всех классов одинакова."""
    counts = np.bincount(labels)
    return 1.0 / counts[labels]


def balanced_sampler(dataset, num_samples: int = None, seed: int = 0) -> WeightedRandomSampler:
    """
    Sampler для DataLoader: за эпоху num_samples примеров (по умолчанию len(dataset)),
    классы примерно поровну. seed — чтобы выбор повторялся от запуска к запуску.
    """
    weights = class_weights(sample_labels(dataset))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=num_samples or len(weights), replacement=True,
                                 generator=generator)
