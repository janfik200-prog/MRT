"""
Разбиение на train/test, чистка и манифест.

    from data_prep.split import build_manifest, save_manifest
    manifest = build_manifest()     # pandas.DataFrame, по строке на снимок
    save_manifest(manifest)         # -> MANIFEST (mri/config.py)

Столбцы манифеста:
    file            путь относительно корня проекта (dataset/<источник>/...)
    source          источник
    original_split  Training или Testing — папка в dataset/
    class, label    класс и его номер в CLASSES
    group           группа почти одинаковых снимков (тот же снимок или соседние срезы серии)
    image           один и тот же снимок (копии) — строже, чем group; лежит внутри одной группы
    split           train, test или excluded
    reason          почему excluded: label_conflict, duplicate, seen_in_training; иначе пусто
    duplicate_of    для reason=duplicate — какой файл оставлен вместо этого

Правила (assign_splits):
    1. Группа со снимками разных классов исключается целиком (label_conflict): неизвестно,
       какой класс верный.
    2. Training -> train: от каждой копии остаётся один файл (KEEP_PRIORITY в mri/config.py),
       остальные — duplicate.
    3. Testing -> test: снимки, чья группа есть в Training, исключаются (seen_in_training) —
       модель видела этот снимок или соседний срез. От каждой копии остаётся один файл.
    Итог: каждый снимок встречается один раз, группы train и test не пересекаются.
"""
import pandas as pd

from mri.config import KEEP_PRIORITY, MANIFEST, SAME_IMAGE_THRESHOLD, SOURCES
from mri.preprocess import prepare

from .duplicates import components, similar_pairs, thumbnails
from .scan import list_images


def keep_one_copy(df: pd.DataFrame) -> pd.Series:
    """Для каждой строки df — файл, который остаётся от её копии (столбец image): сначала без "aug"
    в имени, затем по KEEP_PRIORITY, затем по пути."""
    order = df.assign(_aug=df["file"].str.contains("aug", case=False),
                      _priority=df["source"].map({s: k for k, s in enumerate(KEEP_PRIORITY)}).fillna(len(KEEP_PRIORITY)))
    kept = order.sort_values(["_aug", "_priority", "file"]).drop_duplicates("image").set_index("image")["file"]
    return df["image"].map(kept)


def _drop_copies(df: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Помечает в df лишние копии из part (reason=duplicate) и возвращает part без них."""
    kept = keep_one_copy(part)
    duplicate = part["file"] != kept
    df.loc[part.index[duplicate], "reason"] = "duplicate"
    df.loc[part.index[duplicate], "duplicate_of"] = kept[duplicate]
    return part[~duplicate]


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    df — столбцы file, source, original_split, label, group, image. Возвращает df со столбцами
    split, reason, duplicate_of по правилам из описания модуля.
    """
    df = df.copy()
    df["split"], df["reason"], df["duplicate_of"] = "excluded", "", ""

    # 1. Ошибки разметки — вся группа.
    conflict = df.groupby("group")["label"].transform("nunique") > 1
    df.loc[conflict, "reason"] = "label_conflict"

    # 2. Training: одна копия каждого снимка.
    train = _drop_copies(df, df[(df["original_split"] == "Training") & (df["reason"] == "")])
    df.loc[train.index, "split"] = "train"

    # 3. Testing: без снимков, чья группа есть в Training, и по одной копии каждого снимка.
    test = df[(df["original_split"] == "Testing") & (df["reason"] == "")]
    seen = test["group"].isin(df.loc[df["original_split"] == "Training", "group"])
    df.loc[test.index[seen], "reason"] = "seen_in_training"
    test = _drop_copies(df, test[~seen])
    df.loc[test.index, "split"] = "test"
    return df


def build_manifest(sources=SOURCES) -> pd.DataFrame:
    """
    Все снимки sources: группы и копии, разбиение и чистка (см. описание модуля).
    Похожие снимки ищутся по всем источникам и Training+Testing сразу: копии одного снимка
    лежат в разных источниках и по обе стороны исходного разделения.
    """
    records = list_images("Training", sources) + list_images("Testing", sources)
    i, j, sim = similar_pairs(thumbnails(prepare(r.path) for r in records))
    same = sim > SAME_IMAGE_THRESHOLD
    m = pd.DataFrame({
        "file": [r.file for r in records],
        "source": [r.source for r in records],
        "original_split": [r.split for r in records],
        "class": [r.cls for r in records],
        "label": [r.label for r in records],
        "group": components(len(records), i, j),
        "image": components(len(records), i[same], j[same]),
    })
    return assign_splits(m)


def save_manifest(manifest: pd.DataFrame, path=MANIFEST):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print("Манифест сохранён:", path, f"({len(manifest)} снимков, из них в обучении и оценке "
                                      f"{(manifest['split'] != 'excluded').sum()})")
