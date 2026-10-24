"""
Проверки mri/augment.py: операции, которые повторяют OpenCV, сравниваются с ним,
остальное — по свойствам (размер, тип, повторяемость, сила).

    python -m unittest discover -s tests      # из корня проекта
"""
import unittest

import cv2
import numpy as np
import torch

from mri.augment import (LEVELS, affine, augment, clahe, elastic_field, gaussian_blur, gaussian_kernel,
                         hflip, resize_bilinear, resize_nearest, to_uint8, warp_affine)
from mri.train import MRIDataset


def random_image(seed=0, size=224, smooth=False):
    """float32 size×size со значениями 0..255; smooth — без резких перепадов."""
    x = np.random.default_rng(seed).uniform(0, 255, (size, size)).astype(np.float32)
    if smooth:
        x = cv2.GaussianBlur(x, (0, 0), 3)
        x = (x - x.min()) / (x.max() - x.min()) * 255
    return x


class TestMatchesOpenCV(unittest.TestCase):
    def test_gaussian_kernel(self):
        for sigma, ksize in [(0.5, 3), (1.0, 5), (6.0, 49)]:
            expected = cv2.getGaussianKernel(ksize, sigma)[:, 0]
            np.testing.assert_allclose(gaussian_kernel(sigma, ksize), expected, atol=1e-6)

    def test_gaussian_blur(self):
        x = random_image()
        for sigma, ksize in [(0.5, 3), (1.25, 5)]:
            expected = cv2.GaussianBlur(x, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
            np.testing.assert_allclose(gaussian_blur(x, sigma, ksize), expected, atol=1e-3)

    def test_resize_bilinear(self):
        x = random_image()
        for h, w in [(112, 112), (300, 250), (224, 200)]:
            expected = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_allclose(resize_bilinear(x, h, w), expected, atol=1e-2)

    def test_resize_nearest(self):
        x = random_image()
        for h, w in [(112, 112), (300, 250), (201, 201)]:
            expected = cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(resize_nearest(x, h, w), expected)

    def test_warp_affine(self):
        # OpenCV квантует дробные координаты до 1/32 пикселя, поэтому изображение
        # гладкое и допуск — 1 уровень яркости.
        x = random_image(smooth=True)
        M = cv2.getRotationMatrix2D((111.5, 111.5), 7, 1.03)
        M[:, 2] += (3, -2)
        expected = cv2.warpAffine(x, M, (224, 224), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_allclose(warp_affine(x, M), expected, atol=1.0)

    def test_clahe(self):
        x = to_uint8(random_image(smooth=True))
        for clip in (1.0, 2.0):
            expected = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(x)
            diff = np.abs(to_uint8(clahe(x, clip)).astype(int) - expected)
            self.assertLessEqual(diff.max(), 1)


class TestOperations(unittest.TestCase):
    def test_affine_identity(self):
        x = random_image()
        np.testing.assert_allclose(affine(x), x, atol=1e-3)

    def test_hflip_twice(self):
        x = random_image()
        np.testing.assert_array_equal(hflip(hflip(x)), x)

    def test_elastic_strength(self):
        # Слишком слабое поле не меняет снимок (так было при alpha=30), слишком сильное ломает анатомию.
        rng = np.random.default_rng(0)
        m = np.stack([np.hypot(*elastic_field((224, 224), 80.0, 6.0, rng)) for _ in range(20)])
        self.assertTrue(0.8 < m.mean() < 2.0, m.mean())


class TestAugment(unittest.TestCase):
    def test_shape_and_type(self):
        img = to_uint8(random_image())
        for level in LEVELS:
            out = augment(img, level, np.random.default_rng(0))
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_none_is_identity(self):
        img = to_uint8(random_image())
        np.testing.assert_array_equal(augment(img, "none", np.random.default_rng(0)), img)

    def test_reproducible(self):
        img = to_uint8(random_image())
        for level in LEVELS:
            a = augment(img, level, np.random.default_rng(5))
            b = augment(img, level, np.random.default_rng(5))
            np.testing.assert_array_equal(a, b)

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            augment(to_uint8(random_image()), "C", np.random.default_rng(0))


class TestDatasetStreams(unittest.TestCase):
    def test_sources_get_different_random_streams(self):
        x, y = np.zeros((1, 8, 8), np.uint8), np.zeros(1, int)
        torch.manual_seed(0)
        a, b = MRIDataset(x, y, "R", stream=0), MRIDataset(x, y, "R", stream=1)
        self.assertNotEqual(a.rng().random(), b.rng().random())


if __name__ == "__main__":
    unittest.main()
