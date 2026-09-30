"""
Экспорт расширенного датасета на диск (офлайн-аугментация).

    python augmentation/export.py --level R --copies 5
    python augmentation/export.py --level AB --copies 5 --format jpg --out dataset_augmented

Структура результата:
    <out>/train/<class>/<имя>_orig.png      предобработанный оригинал
    <out>/train/<class>/<имя>_aug{k}.png     k = 1..copies, аугментированные варианты
    <out>/val/<class>/<имя>.png              только предобработка
    <out>/test/<class>/<имя>.png             только предобработка (из dataset/Testing)
    <out>/manifest.csv                       split, class, файл, исходный файл, вариант

Аугментация применяется ТОЛЬКО к train: копии одного снимка никогда не попадают в разные выборки.
val — 15 % от dataset/Training (stratified, seed 42). dataset/ не изменяется.
Дубликаты (keep=0 в _context/duplicates.csv) исключены.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from augment import CLASSES, build_transforms, load_image

ROOT = Path(__file__).resolve().parent.parent
SPLIT_SEED = 42
VAL_FRAC = 0.15


def collect(split: str) -> pd.DataFrame:
    """Файлы dataset/<split> без дубликатов с keep=0."""
    dup = pd.read_csv(ROOT / "_context" / "duplicates.csv")
    drop = {(ROOT / p).resolve() for p in dup.loc[dup.keep == 0, "path"]}
    rows = [{"path": p, "cls": cls}
            for cls in CLASSES for p in sorted((ROOT / "dataset" / split / cls).glob("*.jpg"))
            if p.resolve() not in drop]
    return pd.DataFrame(rows)


def split_frames() -> dict:
    train_all = collect("Training")
    train, val = train_test_split(train_all, test_size=VAL_FRAC, stratify=train_all.cls, random_state=SPLIT_SEED)
    return {"train": train, "val": val, "test": collect("Testing")}


def main(level: str, copies: int, out: Path, fmt: str, seed: int):
    prep = build_transforms("none")
    tf = build_transforms(level)
    tf.set_random_seed(seed)  # воспроизводимый экспорт
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    rows = []
    for split, df in split_frames().items():
        for r in tqdm(df.itertuples(), total=len(df), desc=split):
            d = out / split / r.cls
            d.mkdir(parents=True, exist_ok=True)
            img = prep(image=load_image(r.path))["image"]
            if split == "train":
                variants = [("orig", img)] + [(f"aug{k}", tf(image=img)["image"]) for k in range(1, copies + 1)]
            else:
                variants = [("", img)]
            for tag, im in variants:
                f = d / f"{r.path.stem}{'_' + tag if tag else ''}.{fmt}"
                cv2.imwrite(str(f), im, params)
                rows.append({"split": split, "class": r.cls, "file": f.relative_to(out).as_posix(),
                             "source": r.path.relative_to(ROOT).as_posix(), "variant": tag or "orig"})

    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    (out / "README.txt").write_text(
        f"Расширенный датасет. Аугментация: уровень {level}, {copies} копий на снимок train, seed {seed}.\n"
        f"Аугментирован только train; val и test — только предобработка (grayscale, обрезка полей, 224x224).\n",
        encoding="utf-8")
    print(man.groupby(["split", "class"]).size().unstack(0))
    print("Готово:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="R")
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--format", choices=("jpg", "png"), default="png")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset_augmented")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.level, a.copies, a.out, a.format, a.seed)
