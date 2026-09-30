"""
Превью аугментаций. Ничего не пишет в dataset/, картинки сохраняются в augmentation/preview/.

    python preview.py            # по одному случайному снимку на класс
    python preview.py --seed 7   # другие снимки
"""
import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from augment import (CLASSES, DATASET_DIR, IMG_SIZE, RandomBiasField, RandomGhosting,
                     RandomMotion, build_transforms, load_image)

OUT_DIR = Path(__file__).resolve().parent / "preview"
S = IMG_SIZE

# Каждая аугментация отдельно (p=1), чтобы увидеть её эффект
SINGLE = [
    ("A: HFlip", A.HorizontalFlip(p=1)),
    ("A: Rotate 15", A.Affine(rotate=(15, 15), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)),
    ("A: Shift+Scale", A.Affine(translate_percent=(0.1, 0.1), scale=(1.1, 1.1),
                                border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)),
    ("A: Bright/Contr", A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2),
                                                   contrast_limit=(0.2, 0.2), p=1)),
    ("A: Gamma 0.8", A.RandomGamma(gamma_limit=(80, 80), p=1)),
    ("A: Blur", A.GaussianBlur(blur_limit=(5, 5), p=1)),
    ("B: Noise", A.GaussNoise(std_range=(0.04, 0.04), p=1)),
    ("B: JPEG q=50", A.ImageCompression(quality_range=(50, 50), p=1)),
    ("B: Downscale .5", A.Downscale(scale_range=(0.5, 0.5), p=1)),
    ("C: Bias field", RandomBiasField(coef=0.3, p=1)),
    ("C: Ghosting", RandomGhosting(intensity=(0.2, 0.2), p=1)),
    ("C: Motion", RandomMotion(p=1)),
    ("G: ResizedCrop", A.RandomResizedCrop(size=(S, S), scale=(0.75, 0.75), ratio=(1, 1), p=1)),
    ("G: Elastic", A.ElasticTransform(alpha=30, sigma=6, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)),
    ("G: GridDistort", A.GridDistortion(num_steps=5, distort_limit=(0.15, 0.15),
                                        border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)),
    ("G: OpticalDist", A.OpticalDistortion(distort_limit=(0.1, 0.1), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)),
    ("I: CLAHE", A.CLAHE(clip_limit=(3, 3), p=1)),
    ("I: Sharpen", A.Sharpen(alpha=(0.3, 0.3), p=1)),
    ("I: Unsharp", A.UnsharpMask(alpha=(0.5, 0.5), p=1)),
]


def label(img, text):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    cv2.rectangle(img, (0, 0), (S, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img):
    return build_transforms("none")(image=img)["image"]


def pick(seed):
    rnd = random.Random(seed)
    return {c: rnd.choice(sorted((DATASET_DIR / "Training" / c).glob("*.jpg"))) for c in CLASSES}


def grid(rows):
    return np.vstack([np.hstack(r) for r in rows])


def main(seed):
    OUT_DIR.mkdir(exist_ok=True)
    files = pick(seed)

    # 1) Каждая аугментация по отдельности
    rows = []
    for c, f in files.items():
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        pre = load_image(f)
        row = [label(fit(raw), f"{c}: original"), label(fit(pre), "preprocessed")]
        base = fit(pre)
        for name, t in SINGLE:
            row.append(label(t(image=base.copy())["image"], name))
        rows.append(row)
    # 21 колонка -> три полосы по 7 для удобного просмотра
    bands = [[r[i:i + 7] for r in rows] for i in range(0, len(rows[0]), 7)]
    cv2.imwrite(str(OUT_DIR / "1_single_augs.jpg"), grid(sum(bands, [])), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2) Случайные комбинации по уровням (что реально увидит сеть при обучении)
    for level in ("A", "AB", "ABC", "ABCGI", "R"):
        tf = build_transforms(level)
        rows = []
        for c, f in files.items():
            pre = load_image(f)
            row = [label(fit(pre), f"{c}: preproc")]
            for i in range(6):
                row.append(label(tf(image=pre)["image"], f"{level} #{i + 1}"))
            rows.append(row)
        cv2.imwrite(str(OUT_DIR / f"2_level_{level}.jpg"), grid(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("Снимки:", *[f"  {c}: {f.name}" for c, f in files.items()], sep="\n")
    print("Превью сохранено в", OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
