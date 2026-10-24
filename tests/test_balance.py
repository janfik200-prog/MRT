"""
Проверки mri/balance.py: метки примеров с учётом expand и равные доли классов при выборке.

    python -m unittest discover -s tests      # из корня проекта
"""
import unittest

import numpy as np
from torch.utils.data import ConcatDataset

from mri.balance import balanced_sampler, class_weights, sample_labels
from mri.train import MRIDataset


def part(labels, level="R", expand=3):
    y = np.array(labels)
    return MRIDataset(np.zeros((len(y), 8, 8), np.uint8), y, level, expand)


class TestBalance(unittest.TestCase):
    def test_sample_labels_follow_expand(self):
        data = ConcatDataset([part([0, 1], expand=3), part([2], level="none")])  # none -> expand 1
        np.testing.assert_array_equal(sample_labels(data), [0, 1, 0, 1, 0, 1, 2])
        self.assertEqual(len(sample_labels(data)), len(data))
        self.assertEqual([int(data[i][1]) for i in range(len(data))], list(sample_labels(data)))

    def test_weights_equal_per_class(self):
        labels = np.array([0] * 30 + [1] * 10 + [2] * 5)
        w = class_weights(labels)
        np.testing.assert_allclose([w[labels == c].sum() for c in range(3)], 1.0)

    def test_classes_drawn_equally(self):
        data = ConcatDataset([part([0] * 300 + [1] * 100 + [2] * 50 + [3] * 50)])
        drawn = sample_labels(data)[list(balanced_sampler(data, num_samples=20000, seed=0))]
        share = np.bincount(drawn) / len(drawn)
        np.testing.assert_allclose(share, 0.25, atol=0.02)

    def test_epoch_size_and_reproducible(self):
        data = ConcatDataset([part([0, 0, 0, 1])])
        a, b = list(balanced_sampler(data, seed=3)), list(balanced_sampler(data, seed=3))
        self.assertEqual(len(a), len(data))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
