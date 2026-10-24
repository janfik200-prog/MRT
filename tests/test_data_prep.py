"""
Проверки data_prep/: поиск похожих снимков, разбиение и чистка.

    python -m unittest discover -s tests      # из корня проекта
"""
import unittest

import cv2
import numpy as np
import pandas as pd

from data_prep.duplicates import ORIENTATIONS, components, near_duplicate_groups, similar_pairs, thumbnails
from data_prep.split import assign_splits, keep_one_copy


def brain_like(seed):
    """Случайный «снимок» 224×224: плавные пятна, не похожие на другие seed."""
    noise = np.random.default_rng(seed).uniform(0, 255, (14, 14)).astype(np.float32)
    return cv2.resize(noise, (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)


class TestDuplicates(unittest.TestCase):
    def test_variants_grouped_distinct_separate(self):
        a, b = brain_like(1), brain_like(2)
        noisy = np.clip(a + np.random.default_rng(0).normal(0, 8, a.shape), 0, 255).astype(np.uint8)
        brighter = np.clip(a.astype(int) * 1.1 + 10, 0, 255).astype(np.uint8)
        images = [a, a[:, ::-1], a[::-1], a[::-1, ::-1], noisy, brighter, b]
        g = near_duplicate_groups(thumbnails(images))
        self.assertEqual(len(set(g[:6])), 1, g)  # все варианты a — одна группа
        self.assertNotEqual(g[6], g[0])          # b — отдельно

    def test_chain(self):
        # a≈b и b≈c (сдвиг на 2 пикселя) — одна группа, даже если a и c различаются сильнее.
        a = brain_like(3)
        b, c = np.roll(a, 2, axis=1), np.roll(a, 4, axis=1)
        g = near_duplicate_groups(thumbnails([a, b, c, brain_like(4)]))
        self.assertEqual(len(set(g[:3])), 1, g)
        self.assertNotEqual(g[3], g[0])

    def test_pairs_have_similarity_and_no_self(self):
        a = brain_like(5)
        i, j, sim = similar_pairs(thumbnails([a, a[:, ::-1], brain_like(6)]))
        self.assertFalse(np.any(i == j))
        self.assertEqual({(0, 1), (1, 0)}, set(zip(i.tolist(), j.tolist())))
        self.assertTrue(np.all(sim > 0.99))

    def test_components(self):
        g = components(5, np.array([0, 1]), np.array([1, 2]))  # 0-1-2 вместе, 3 и 4 по одному
        self.assertEqual(len({g[0], g[1], g[2]}), 1)
        self.assertEqual(len(set(g)), 3)

    def test_flip_thumbnail_equals_thumbnail_of_flip(self):
        # 224 / 32 = 7: уменьшение — среднее блоков 7×7, поэтому отражать можно уже уменьшенную копию.
        a = np.random.default_rng(5).integers(0, 256, (224, 224), dtype=np.uint8)
        for transform in ORIENTATIONS.values():
            np.testing.assert_allclose(transform(thumbnails([a]))[0], thumbnails([transform(a)])[0], atol=1e-4)


def table(rows):
    """DataFrame из (file, source, original_split, label, group, image)."""
    return pd.DataFrame(rows, columns=["file", "source", "original_split", "label", "group", "image"])


class TestKeepOneCopy(unittest.TestCase):
    def test_priority(self):
        df = table([
            ("dataset/BDNeuro/a.jpg", "BDNeuro", "Training", 0, 1, 1),
            ("dataset/KaggleBrainTumor/Tr-aug-a.jpg", "KaggleBrainTumor", "Training", 0, 1, 1),
            ("dataset/KaggleBrainTumor/Tr-a.jpg", "KaggleBrainTumor", "Training", 0, 1, 1),
            ("dataset/BDNeuro/b.jpg", "BDNeuro", "Training", 0, 1, 2),
            ("dataset/BDNeuro/b2.jpg", "BDNeuro", "Training", 0, 1, 2),
        ])
        kept = keep_one_copy(df)
        # без "aug" и не BDNeuro; если все из BDNeuro — по пути
        self.assertEqual(set(kept[:3]), {"dataset/KaggleBrainTumor/Tr-a.jpg"})
        self.assertEqual(set(kept[3:]), {"dataset/BDNeuro/b.jpg"})


class TestAssignSplits(unittest.TestCase):
    def setUp(self):
        self.df = table([
            # группа 1: снимок 10 в двух копиях в Training и его копия в Testing -> test исключается
            ("dataset/PMRAM/Training/x.jpg", "PMRAM", "Training", 0, 1, 10),
            ("dataset/BDNeuro/Training/x.jpg", "BDNeuro", "Training", 0, 1, 10),
            ("dataset/BDNeuro/Testing/x.jpg", "BDNeuro", "Testing", 0, 1, 10),
            # группа 2: разные классы -> исключается целиком
            ("dataset/PMRAM/Training/y.jpg", "PMRAM", "Training", 0, 2, 20),
            ("dataset/PMRAM/Testing/y.jpg", "PMRAM", "Testing", 1, 2, 21),
            # группа 3: только в Testing, две копии -> одна в test
            ("dataset/PMRAM/Testing/z.jpg", "PMRAM", "Testing", 1, 3, 30),
            ("dataset/BDNeuro/Testing/z.jpg", "BDNeuro", "Testing", 1, 3, 30),
            # группа 4: только в Training
            ("dataset/PMRAM/Training/w.jpg", "PMRAM", "Training", 1, 4, 40),
        ])
        self.out = assign_splits(self.df).set_index("file")

    def test_reasons(self):
        reason = self.out["reason"]
        self.assertEqual(reason["dataset/BDNeuro/Training/x.jpg"], "duplicate")
        self.assertEqual(self.out.loc["dataset/BDNeuro/Training/x.jpg", "duplicate_of"],
                         "dataset/PMRAM/Training/x.jpg")
        self.assertEqual(reason["dataset/BDNeuro/Testing/x.jpg"], "seen_in_training")
        self.assertEqual(reason["dataset/PMRAM/Training/y.jpg"], "label_conflict")
        self.assertEqual(reason["dataset/PMRAM/Testing/y.jpg"], "label_conflict")
        self.assertEqual(reason["dataset/BDNeuro/Testing/z.jpg"], "duplicate")

    def test_splits(self):
        split = self.out["split"]
        self.assertEqual(split["dataset/PMRAM/Training/x.jpg"], "train")
        self.assertEqual(split["dataset/PMRAM/Training/w.jpg"], "train")
        self.assertEqual(split["dataset/PMRAM/Testing/z.jpg"], "test")
        self.assertEqual((split == "excluded").sum(), 5)
        self.assertTrue((self.out["reason"][split == "excluded"] != "").all())

    def test_each_image_once_and_groups_disjoint(self):
        used = self.out[self.out["split"] != "excluded"]
        self.assertTrue((used.groupby("image").size() == 1).all())
        self.assertTrue((used.groupby("group")["split"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
