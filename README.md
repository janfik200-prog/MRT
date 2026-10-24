# Классификация МРТ головного мозга

Обучение CNN для четырёх классов: глиома (`glioma`), менингиома
(`meningioma`), отсутствие опухоли (`notumor`), опухоль гипофиза (`pituitary`).
В ноутбуке рассчитываются precision и recall по каждому классу (цель — не ниже 0.85).

## Состав

```text
dataset/                 датасет: снимки из четырёх источников (см. ниже)
mri/                     код проекта
  config.py              пути, классы, источники и уровень аугментации для каждого
  preprocess.py          grayscale, обрезка полей, размер 224×224
  augment.py             аугментации на NumPy: none, A, B, AB, R
  data.py                поиск снимков в dataset/ и загрузка в память
  model.py               свёрточная сеть SimpleCNN
  train.py               Dataset для PyTorch, обучение, предсказания, метрики
tests/                   проверки аугментации (сравнение с OpenCV и свойства)
model.ipynb              запуск обучения и таблицы результатов
_work/                   всё создаваемое при работе; в git не попадает (см. ниже)
requirements.txt         зависимости
DATASET_LINKS.txt        источники данных и лицензии
```

Всё, что создаётся при работе — результаты обучения, таблицы и
временные проверки — складывается только в `_work/`. Папка исключена из git
и может быть удалена целиком; в корне проекта и в `dataset/` мусор не появляется.

Все настройки, общие для кода, — в `mri/config.py`. Параметры запуска обучения
(эпохи, batch, seed) — в первой ячейке `model.ipynb`.

## Данные

Датасет один, но собран из четырёх источников — по папке на каждый:

```text
dataset/<источник>/
  Training/<класс>/
  Testing/<класс>/
```

Источники и уровень аугментации для их Training задаются словарём `SOURCES`
в `mri/config.py`: BDNeuro — без аугментации, PMRAM, MultiClass и
KaggleBrainTumor — уровень R. Чтобы добавить источник, положите его папку
в `dataset/` и допишите строку в `SOURCES`.

В проверенной локальной версии:

| Источник | Training | Testing | Всего |
|---|---:|---:|---:|
| PMRAM | 1203 | 302 | 1505 |
| MultiClass | 910 | 227 | 1137 |
| BDNeuro | 5052 | 889 | 5941 |
| KaggleBrainTumor (версия 2) | 5600 | 1600 | 7200 |
| **Всего** | **12765** | **3018** | **15783** |

PMRAM и MultiClass разделены примерно 80/20 по классам, seed 42;
точные копии сгруппированы, но все файлы сохранены.
У BDNeuro исходные train и val объединены в Training, исходный test — в Testing.
У KaggleBrainTumor сохранены исходные Training/Testing версии 2 без переразбиения.

Проверка SHA-256 после добавления выявила 3315 групп точных совпадений между
источниками и 900 групп с участием KaggleBrainTumor, пересекающих Training/Testing.
Дубли ещё не очищены (в PMRAM — 95 дополнительных файлов точных копий).
Между PMRAM и BDNeuro найдены две пары одинаковых grayscale-изображений;
одна пара пересекает PMRAM/Training и BDNeuro/Testing.
Идентификаторы пациентов недоступны, поэтому метрики на Testing пока предварительные.

## Установка и обучение

Команды выполняются из корня проекта. Нужен Python 3.10 или новее.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Откройте `model.ipynb` и выполните ячейки по порядку. Одна CNN обучается
на Training всех источников (BDNeuro — expand 1, остальные — уровень R, expand 3),
30 эпох; Testing оценивается после обучения.
Результаты, веса и история сохраняются в `_work/results/<дата_время>/`:
`config.json`, `history.csv`, `model.pt`, `predictions.csv`, `summary.csv`,
`metrics_by_class.csv`, `confusion_matrix.csv`.
При ошибке рабочих процессов DataLoader на Windows поставьте `NUM_WORKERS = 0`.

Проверки аугментации (около секунды):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Лицензии исходных датасетов и ссылки на авторов указаны в `DATASET_LINKS.txt`.
