# Brain Tumor MRI — классификация опухолей мозга

Классификация МРТ-снимков мозга на 4 класса: `glioma`, `meningioma`, `notumor`, `pituitary`.
Модель — свёрточная сеть, обучаемая с нуля. Цель: **Recall ≥ 85 %** и **Precision ≥ 85 %** по каждому классу.

## Структура

| Путь | Что это |
|---|---|
| `augmentation/augment.py` | предобработка (grayscale, обрезка полей, 224×224) и аугментации уровней `none / A / AB / ABC` |
| `augmentation/dataset.py` | `MRIDataset` для PyTorch: расширение train аугментацией на лету (`expand`) |
| `augmentation/preview.py` | превью аугментаций → `augmentation/preview/` |
| `notebooks/01_simple_cnn.ipynb` | простая CNN, обучение, метрики (Precision / Recall / F1, матрица ошибок) |
| `notebooks/results.csv` | итоги запусков для сравнения уровней аугментации |
| `_context/duplicates.csv` | точные дубликаты в датасете (`keep=0` — исключаются) |
| `photo_2026-09-30_20-12-03.jpg` | схема плана проекта |

## Данные

Датасет в репозиторий не входит. Скачайте [Brain Tumor MRI Dataset (Kaggle, Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) и распакуйте так:

```
dataset/
  Training/{glioma,meningioma,notumor,pituitary}/*.jpg
  Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Затем откройте `notebooks/01_simple_cnn.ipynb` и выберите ядро из `.venv`.

## Аугментация

- **A** — отражение, поворот ±15°, сдвиг, масштаб, яркость/контраст, гамма, размытие.
- **B** — шум, JPEG-сжатие, понижение разрешения (против «подсказок» формата файла в классе `notumor`).
- **C** — МРТ-артефакты: неоднородность поля, ghosting, движение.

Аугментация применяется только к train и на лету: `EXPAND=3` — каждый снимок встречается 3 раза за эпоху, каждый раз в новом варианте.

## Результаты (test, 1584 снимка, 20 эпох)

| Аугментация | accuracy | macro P | macro R | R glioma | R meningioma | R notumor | R pituitary |
|---|---|---|---|---|---|---|---|
| none | 0.921 | 0.929 | 0.920 | 0.741 | 0.950 | 1.000 | 0.988 |
| AB ×3 | 0.930 | 0.933 | 0.929 | 0.798 | 0.930 | 0.998 | 0.990 |

Пока не достигнут Recall ≥ 85 % для `glioma` (путается с `meningioma`).

## Планы

- больше эпох, сравнение уровней `AB` / `ABC`;
- расширение датасета внешними наборами (PMRAM, BRISC 2025 и др.) с проверкой пересечений (MD5 + pHash);
- разметка новых данных.
