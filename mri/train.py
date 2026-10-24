"""
Обучение и оценка: Dataset для PyTorch, цикл обучения, предсказания, метрики, сохранение.

Используется из model.ipynb. Всё, что сохраняется, кладётся в results/<дата_время>/.
"""
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import ConcatDataset, Dataset

from .augment import LEVELS, augment
from .config import CLASSES, RESULTS_DIR, SOURCES, TARGET
from .data import load_images, read_manifest


# ============================================================================
# Данные
# ============================================================================
class MRIDataset(Dataset):
    """
    x: uint8 [N, 224, 224] (после prepare()), y: метки [N].
    level — уровень аугментации (mri/augment.py), "none" — без неё.
    expand > 1: каждый снимок встречается expand раз за эпоху, каждый раз с новой аугментацией.
    stream — номер набора: у частей одного ConcatDataset разные потоки случайных чисел.
    Выход: float32 [1, 224, 224] в диапазоне [-1, 1] и метка.

    Класс лежит в модуле, а не в ноутбуке, чтобы DataLoader(num_workers>0) работал
    на Windows: дочерние процессы должны его импортировать.
    """

    def __init__(self, x, y, level: str = "none", expand: int = 1, stream: int = 0):
        if level not in LEVELS:
            raise ValueError(f"level: один из {LEVELS}, получено {level!r}")
        self.x, self.y, self.level = x, y, level
        self.expand = expand if level != "none" else 1
        self.stream = stream
        self._seed, self._rng = None, None

    def rng(self) -> np.random.Generator:
        # Генератор случайных чисел создаём от torch.initial_seed(): он свой в каждом
        # процессе DataLoader (без persistent_workers — новый каждую эпоху). Иначе все
        # процессы получили бы копию одного генератора и выдавали одинаковые варианты.
        # stream добавляется к seed, чтобы у разных источников потоки не совпадали.
        seed = torch.initial_seed()
        if seed != self._seed:
            self._seed, self._rng = seed, np.random.default_rng([seed, self.stream])
        return self._rng

    def __len__(self):
        return len(self.x) * self.expand

    def __getitem__(self, i):
        i %= len(self.x)
        img = self.x[i] if self.level == "none" else augment(self.x[i], self.level, self.rng())
        t = torch.from_numpy(img).float().div_(127.5).sub_(1.0)[None]  # 0..255 -> -1..1
        return t, int(self.y[i])


def build_datasets(expand: int, policies: dict = SOURCES, manifest: pd.DataFrame = None):
    """
    Наборы по манифесту (data_prep/dataset.ipynb): train — каждый источник со своим уровнем
    аугментации, test — без аугментации. Снимки split=excluded не используются.
    Возвращает: train, test (ConcatDataset),
    test_table — по строке на тестовый снимок (dataset, file, true_index), counts — сводка.
    """
    m = read_manifest() if manifest is None else manifest
    missing = set(policies) - set(m["source"])
    if missing:
        raise ValueError(f"В манифесте нет источников {sorted(missing)}: пересоздайте его в data_prep/dataset.ipynb")

    train_parts, test_parts, test_rows, counts = [], [], [], []
    for stream, (source, level) in enumerate(policies.items()):
        rows = m[m["source"] == source]
        train, test = rows[rows["split"] == "train"], rows[rows["split"] == "test"]
        part = MRIDataset(load_images(train["file"]), train["label"].to_numpy(), level, expand, stream)
        train_parts.append(part)
        test_parts.append(MRIDataset(load_images(test["file"]), test["label"].to_numpy()))
        test_rows.append(test)

        counts.append({"dataset": source, "train_images": len(train), "test_images": len(test),
                       "excluded": int(rows["split"].eq("excluded").sum()),
                       "augmentation": level, "expand": part.expand, "samples_per_epoch": len(part)})
        print(counts[-1], flush=True)

    test_table = (pd.concat(test_rows)[["source", "file", "label"]]
                  .rename(columns={"source": "dataset", "label": "true_index"}).reset_index(drop=True))
    return ConcatDataset(train_parts), ConcatDataset(test_parts), test_table, counts


# ============================================================================
# Обучение
# ============================================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def new_run_dir():
    """Папка results/<ГГГГММДД_ЧЧММСС> (московское время) для одного запуска."""
    run = RESULTS_DIR / datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    return run


def save_json(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fit(model, train_dl, epochs: int, device: str, run_dir, config: dict, lr: float = 1e-3,
        weight_decay: float = 1e-2) -> pd.DataFrame:
    """
    AdamW + OneCycle, на GPU — смешанная точность (AMP).
    После каждой эпохи в run_dir пишутся history.csv и model.pt (модель, оптимизатор, расписание).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=amp):
                out = model(xb)
                loss = loss_fn(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        row = {"epoch": epoch, "loss": loss_sum / total, "train_accuracy": correct / total,
               "seconds": time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "config": config}, run_dir / "model.pt")
        print(f"Эпоха {epoch:2d}/{epochs}: loss {row['loss']:.4f}, "
              f"train accuracy {row['train_accuracy']:.3f}, {row['seconds']:.1f} с", flush=True)
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, dl, device: str) -> np.ndarray:
    """Номер предсказанного класса для каждого снимка dl (в порядке dl)."""
    model.eval()
    pred = []
    for xb, _ in dl:
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out = model(xb.to(device))
        pred.append(out.argmax(1).cpu())
    return torch.cat(pred).numpy()


# ============================================================================
# Метрики
# ============================================================================
def evaluate(test_table: pd.DataFrame, pred: np.ndarray, run_dir=None):
    """
    Precision и recall по классам — для всего теста ("all") и для каждого источника.
    Возвращает (summary, by_class, confusion); если задан run_dir — сохраняет туда
    predictions.csv, summary.csv, metrics_by_class.csv, confusion_matrix.csv.
    """
    y_true = test_table["true_index"].to_numpy()
    labels = range(len(CLASSES))
    masks = {"all": np.ones(len(y_true), bool)}
    masks |= {source: test_table["dataset"].eq(source).to_numpy() for source in test_table["dataset"].unique()}
    rows, summaries = [], []
    for scope, mask in masks.items():
        truth, guess = y_true[mask], pred[mask]
        precision = precision_score(truth, guess, labels=labels, average=None, zero_division=0)
        recall = recall_score(truth, guess, labels=labels, average=None, zero_division=0)
        for i, cls in enumerate(CLASSES):
            rows.append({"scope": scope, "class": cls, "support": int((truth == i).sum()),
                         "precision": float(precision[i]), "recall": float(recall[i]),
                         "target_met": bool(precision[i] >= TARGET and recall[i] >= TARGET)})
        summaries.append({"scope": scope, "images": int(mask.sum()), "accuracy": accuracy_score(truth, guess),
                          "macro_precision": precision.mean(), "macro_recall": recall.mean(),
                          "all_classes_target_met": bool((precision >= TARGET).all() and (recall >= TARGET).all())})
    summary, by_class = pd.DataFrame(summaries), pd.DataFrame(rows)
    confusion = pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=CLASSES, columns=CLASSES)

    if run_dir is not None:
        predictions = test_table.assign(true_class=[CLASSES[i] for i in y_true],
                                        predicted_class=[CLASSES[i] for i in pred])
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        summary.to_csv(run_dir / "summary.csv", index=False)
        by_class.to_csv(run_dir / "metrics_by_class.csv", index=False)
        confusion.to_csv(run_dir / "confusion_matrix.csv")
    return summary, by_class, confusion
