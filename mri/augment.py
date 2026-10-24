"""
Аугментация снимков МРТ на чистом numpy.

Картинка — просто матрица чисел: x[строка, столбец] = яркость 0..255.
Разделы 1–5 — операции: обычные функции «матрица -> матрица» с явными параметрами.
Раздел 6 — уровни A/B/R: случайно решают, какие операции применить и с какой силой.

    import numpy as np
    from mri.preprocess import prepare
    from mri.augment import augment

    rng = np.random.default_rng(0)        # один и тот же seed — те же самые варианты
    img = prepare(path)                   # uint8 224x224
    out = augment(img, "AB", rng)         # каждый вызов — новый случайный вариант

Уровни (точные диапазоны — в aug_A, aug_B и R_POOL ниже):
    none — без изменений
    A    — отражение, поворот ±7°, сдвиг ±2.5 %, масштаб ±3 %, яркость/контраст, гамма, размытие
    B    — шум, JPEG-сжатие, понижение разрешения (против подсказок по формату файла)
    AB   — сначала A, потом B
    R    — RandAugment-подобная: отражение + 2 случайные операции из пула R_POOL
           (геометрия, яркость, качество файла)

Всё на numpy, кроме одного места: jpeg() вызывает кодек OpenCV — формат JPEG
занимает сотни строк, переписывать его незачем.
Диапазоны случайных параметров смягчены для сохранения деталей снимка.
"""
import cv2  # только для JPEG-кодека в jpeg()
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================================
# 0. Типы данных
# ============================================================================
# Снимок хранится как uint8: целые 0..255. При переполнении uint8 «заворачивается»:
# 250 + 10 = 4 (светлый пиксель стал бы чёрным). Поэтому считаем во float32,
# операции, которые могут выйти за 0..255, сами обрезают результат (np.clip),
# а в конце возвращаем uint8.
def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)  # rint — округление до целого


# ============================================================================
# 1. Геометрия: откуда брать пиксели
# ============================================================================
# Все геометрические операции устроены одинаково: для каждой ячейки НОВОЙ картинки
# считаем, из какой точки (sx, sy) СТАРОЙ картинки взять значение, и берём его
# функцией bilinear(). Если двигать пиксели «вперёд», после округления часть
# ячеек останется пустой (дырки), поэтому всегда идём «назад».

def bilinear(x: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """
    Значения картинки x в дробных точках (sx, sy). Точки за краем — 0 (чёрный).

    Точка (4.3, 7.6) лежит между 4 пикселями: (4, 7), (5, 7), (4, 8), (5, 8).
    Берём их среднее с весами: чем ближе пиксель к точке, тем больше его вес.
    """
    h, w = x.shape
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)  # левый верхний сосед
    dx, dy = sx - x0, sy - y0                                    # дробные части 0..1

    def at(yy, xx):  # значение пикселя (yy, xx), а за краем картинки — 0
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        return np.where(inside, x[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)], 0)

    return (at(y0, x0) * (1 - dx) * (1 - dy) + at(y0, x0 + 1) * dx * (1 - dy)
            + at(y0 + 1, x0) * (1 - dx) * dy + at(y0 + 1, x0 + 1) * dx * dy)


def grid(h: int, w: int):
    """Координаты всех ячеек: xs[i, j] = j (номер столбца), ys[i, j] = i (номер строки)."""
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.float32), ys.astype(np.float32)


def warp_affine(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Аффинное преобразование: каждая точка (x, y) переезжает в M · (x, y, 1).
    M — матрица 2×3 (поворот/масштаб/скос в левой части 2×2, сдвиг в правом столбце).
    Чтобы идти «назад», обращаем матрицу: старая точка = M⁻¹ · новая.
    """
    h, w = x.shape
    M3 = np.vstack([M, [0, 0, 1]])        # 3×3, чтобы матрицу можно было обратить
    inv = np.linalg.inv(M3)
    xs, ys = grid(h, w)
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return bilinear(x, sx, sy)


def affine(x: np.ndarray, angle: float = 0.0, scale_x: float = 1.0, scale_y: float = 1.0,
           tx: float = 0.0, ty: float = 0.0, shear_x: float = 0.0, shear_y: float = 0.0) -> np.ndarray:
    """
    Поворот на angle градусов (против часовой стрелки), масштаб, скос (shear, градусы)
    вокруг центра картинки, затем сдвиг на (tx, ty) пикселей. Открывшиеся углы — чёрные.
    Матрица собирается так же, как в albumentations: сначала масштаб, потом поворот,
    потом скос, потом сдвиг.
    """
    h, w = x.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    a = np.deg2rad(angle)
    # Ось y в картинке направлена ВНИЗ, поэтому знак при sin обратный обычному:
    # так положительный угол даёт поворот против часовой стрелки на экране.
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    S = np.diag([scale_x, scale_y])
    Sh = np.array([[1, np.tan(np.deg2rad(shear_x))], [np.tan(np.deg2rad(shear_y)), 1]])
    A2 = Sh @ R @ S                                   # итоговая матрица 2×2
    c = np.array([cx, cy])
    # новая = A2 · (старая − центр) + центр + сдвиг
    shift = c + np.array([tx, ty]) - A2 @ c
    return warp_affine(x, np.hstack([A2, shift[:, None]]))


def resize_bilinear(x: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Изменение размера с билинейной интерполяцией (как cv2.INTER_LINEAR).
    Центр новой ячейки j соответствует точке (j + 0.5) · (w / new_w) − 0.5 старой.
    За край не выходим: координаты прижимаем к границам (повтор крайних пикселей).
    """
    h, w = x.shape
    sx = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    sy = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
    SX, SY = np.meshgrid(sx, sy)
    return bilinear(x, SX, SY)


def resize_nearest(x: np.ndarray, new_h: int, new_w: int, fy: float = None, fx: float = None) -> np.ndarray:
    """
    Изменение размера «ближайшим соседом» (как cv2.INTER_NEAREST): каждая новая ячейка
    просто копирует один старый пиксель — отсюда «квадратики» при увеличении.
    fx, fy — коэффициенты масштаба, если они заданы явно (иначе new / old).
    """
    h, w = x.shape
    fx = fx or new_w / w
    fy = fy or new_h / h
    cols = np.minimum(np.floor(np.arange(new_w) / fx).astype(int), w - 1)
    rows = np.minimum(np.floor(np.arange(new_h) / fy).astype(int), h - 1)
    return x[np.ix_(rows, cols)]  # выбрать нужные строки и столбцы


# ============================================================================
# 2. Яркость
# ============================================================================
def hflip(x: np.ndarray) -> np.ndarray:
    """Зеркало слева направо: столбцы в обратном порядке."""
    return x[:, ::-1]


def brightness_contrast(x: np.ndarray, brightness: float = 0.0, contrast: float = 0.0) -> np.ndarray:
    """
    Как в albumentations: x · (1 + contrast) + brightness · 255.
    contrast=0.2 — всё умножить на 1.2; brightness=0.1 — ко всему прибавить 25.5.
    Прибавка идёт и к чёрному фону — поэтому после неё фон становится серым.
    """
    return np.clip(x * (1 + contrast) + brightness * 255, 0, 255)


def gamma(x: np.ndarray, g: float = 1.0) -> np.ndarray:
    """
    Гамма: y = 255 · (x/255)^g. Чёрное (0) и белое (255) не меняются,
    g < 1 осветляет тёмные области, g > 1 затемняет.
    """
    return 255 * (x / 255) ** g


# ============================================================================
# 3. Свёртки: размытие и резкость
# ============================================================================
# Свёртка: новый пиксель = сумма соседей × веса (ядро). Для каждого пикселя
# берём «окно» соседей и считаем скалярное произведение окна на вектор весов.
# sliding_window_view делает из картинки h×w массив окон h×w×k без копирования.

def gaussian_kernel(sigma: float, ksize: int) -> np.ndarray:
    """
    Веса соседей по колоколу Гаусса, например для ksize=5: [0.06, 0.24, 0.40, 0.24, 0.06].
    sigma <= 0 — подобрать по размеру ядра (формула OpenCV).
    """
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    r = np.arange(ksize) - ksize // 2      # -2, -1, 0, 1, 2
    k = np.exp(-0.5 * (r / sigma) ** 2)
    return k / k.sum()                     # сумма весов = 1, средняя яркость не меняется


def convolve_1d(x: np.ndarray, k: np.ndarray, axis: int, edge: str = "reflect") -> np.ndarray:
    """Свёртка вдоль одной оси (1 — по строкам, 0 — по столбцам)."""
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode=edge)  # на краях: "reflect" — зеркально, "edge" — повтор крайнего
    windows = sliding_window_view(xp, len(k), axis=axis)  # h × w × len(k)
    return windows @ k                                    # скалярное произведение окна на веса


def gaussian_blur(x: np.ndarray, sigma: float = 1.0, ksize: int = 5, edge: str = "reflect") -> np.ndarray:
    """
    Каждый пиксель заменяем средним соседей с весами gaussian_kernel.
    Гауссово ядро «разделимо»: размыть по строкам, потом по столбцам —
    то же самое, что сразу квадратом ksize×ksize, но быстрее.
    """
    k = gaussian_kernel(sigma, ksize)
    return convolve_1d(convolve_1d(x, k, axis=1, edge=edge), k, axis=0, edge=edge)


def convolve_2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Свёртка с квадратным ядром (например 3×3): окно соседей × ядро, сумма."""
    r = kernel.shape[0] // 2
    windows = sliding_window_view(np.pad(x, r, mode="reflect"), kernel.shape)  # h × w × 3 × 3
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sharpen(x: np.ndarray, alpha: float = 0.2, lightness: float = 1.0) -> np.ndarray:
    """
    Резкость: ядро 3×3 «центр минус соседи» подчёркивает перепады яркости (края).
    alpha — сила (0 — без изменений), lightness — насколько осветлить центр.
    """
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    effect = np.array([[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]], np.float32)
    return np.clip(convolve_2d(x, (1 - alpha) * identity + alpha * effect), 0, 255)


def clahe(x: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """
    CLAHE — выравнивание контраста по участкам (как cv2.createCLAHE).

    1. Делим снимок на tiles×tiles плиток и для каждой строим гистограмму яркостей.
    2. Обрезаем высокие столбцы гистограммы (clip_limit), излишек раздаём поровну —
       иначе однородные области (фон) получили бы огромный контраст и шум.
    3. Накопленная гистограмма плитки — таблица «старая яркость -> новая».
    4. Чтобы не было ступенек на границах плиток, для каждого пикселя смешиваем
       таблицы 4 ближайших плиток (та же билинейная идея, что и в bilinear()).
    """
    img = to_uint8(x)
    h, w = img.shape
    th, tw = -(-h // tiles), -(-w // tiles)                       # размер плитки (округление вверх)
    padded = np.pad(img, ((0, th * tiles - h), (0, tw * tiles - w)), mode="reflect")
    area = th * tw

    # 1. Гистограммы всех плиток сразу: [tiles, tiles, 256]
    blocks = padded.reshape(tiles, th, tiles, tw).transpose(0, 2, 1, 3).reshape(tiles * tiles, area)
    offset = 256 * np.arange(tiles * tiles)[:, None]              # своя «полка» номеров для каждой плитки
    hist = np.bincount((blocks + offset).ravel(), minlength=256 * tiles * tiles).reshape(tiles * tiles, 256)

    # 2. Обрезка и раздача излишка
    limit = max(int(clip_limit * area / 256), 1)
    excess = np.maximum(hist - limit, 0).sum(1)
    hist = np.minimum(hist, limit) + (excess // 256)[:, None]
    residual = excess % 256                                       # остаток — по 1 в каждый step-й столбец
    step = np.maximum(256 // np.maximum(residual, 1), 1)
    bins = np.arange(256)
    hist += (bins % step[:, None] == 0) & (bins // step[:, None] < residual[:, None])

    # 3. Таблицы перекодировки
    lut = np.clip(np.rint(np.cumsum(hist, 1) * 255 / area), 0, 255).reshape(tiles, tiles, 256)

    # 4. Смешивание таблиц 4 ближайших плиток
    def neighbours(n, size):  # для координаты: номер плитки слева/справа и вес правой
        f = np.arange(n) / size - 0.5
        t1 = np.floor(f).astype(int)
        return np.maximum(t1, 0), np.minimum(t1 + 1, tiles - 1), (f - t1)

    y1, y2, wy = neighbours(h, th)
    x1, x2, wx = neighbours(w, tw)
    Y1, Y2, X1, X2 = y1[:, None], y2[:, None], x1[None, :], x2[None, :]
    top = lut[Y1, X1, img] * (1 - wx) + lut[Y1, X2, img] * wx
    bottom = lut[Y2, X1, img] * (1 - wx) + lut[Y2, X2, img] * wx
    return top * (1 - wy[:, None]) + bottom * wy[:, None]


# ============================================================================
# 4. Шум и качество файла
# ============================================================================
def gauss_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """К каждому пикселю — своё случайное число из N(0, std·255)."""
    return np.clip(x + rng.normal(0, std * 255, x.shape), 0, 255)


def jpeg(x: np.ndarray, quality: int) -> np.ndarray:
    """Сохранить в JPEG с качеством quality и прочитать обратно (кодек OpenCV)."""
    _, buf = cv2.imencode(".jpg", to_uint8(x), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def downscale(x: np.ndarray, scale: float) -> np.ndarray:
    """
    Уменьшить в scale раз и увеличить обратно — теряются мелкие детали.
    Ближайший сосед, как в albumentations по умолчанию (отсюда «квадратики»).
    """
    h, w = x.shape
    small = resize_nearest(x, round(h * scale), round(w * scale), fy=scale, fx=scale)
    return resize_nearest(small, h, w)


# ============================================================================
# 5. Дополнительная геометрия
# ============================================================================
def crop_resize(x: np.ndarray, top: int, left: int, ch: int, cw: int) -> np.ndarray:
    """Вырезать прямоугольник ch×cw с углом (top, left) и растянуть обратно до исходного размера."""
    h, w = x.shape
    return resize_bilinear(x[top:top + ch, left:left + cw], h, w)


def elastic(x: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Эластичная деформация: каждый пиксель берём из точки, смещённой на (dx, dy)."""
    xs, ys = grid(*x.shape)
    return bilinear(x, xs + dx, ys + dy)


def elastic_field(shape, alpha: float, sigma: float, rng: np.random.Generator):
    """
    Случайное плавное поле смещений (как в albumentations): шум N(0, 1), делённый
    на свой максимум, размытый гауссом с sigma и умноженный на alpha.
    """
    f = rng.standard_normal((2, *shape)).astype(np.float32)
    f /= np.abs(f).max()
    ksize = int(round(sigma * 8 + 1)) | 1                  # размер ядра по правилу OpenCV
    f = [gaussian_blur(fi, sigma, ksize, edge="edge") * alpha for fi in f]
    return f[0], f[1]


# ============================================================================
# 6. Уровни: случайный выбор операций и параметров
# ============================================================================
# Каждая операция применяется с вероятностью p: «бросаем монетку» rng.random() < p.
# Вероятности операций сохранены; диапазоны случайных параметров смягчены.

def _blur_ksize(rng, lo=3, hi=5):
    k = int(rng.integers(lo, hi + 1))   # как в albumentations: 3, 4 или 5; чётное -> +1
    return k + 1 if k % 2 == 0 else k


def rand_affine(x, rng, rotate=0.0, shift=0.0, scale=0.0, shear=0.0):
    """Случайное аффинное: поворот ±rotate°, сдвиг ±shift доли размера, масштаб 1±scale, скос ±shear°."""
    h, w = x.shape
    return affine(x,
                  angle=rng.uniform(-rotate, rotate),
                  scale_x=rng.uniform(1 - scale, 1 + scale),   # по x и y — независимо
                  scale_y=rng.uniform(1 - scale, 1 + scale),
                  tx=round(rng.uniform(-shift, shift) * w),    # albumentations округляет сдвиг до пикселей
                  ty=round(rng.uniform(-shift, shift) * h),
                  shear_x=rng.uniform(-shear, shear),
                  shear_y=rng.uniform(-shear, shear))


def rand_crop(x, rng, scale=(0.95, 1.0), ratio=(0.97, 1.03)):
    """Случайный прямоугольник площадью scale от снимка и соотношением сторон ratio."""
    h, w = x.shape
    for _ in range(10):  # 10 попыток подобрать прямоугольник, который помещается в снимок
        area = rng.uniform(*scale) * h * w
        r = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw, ch = round(np.sqrt(area * r)), round(np.sqrt(area / r))
        if 0 < cw <= w and 0 < ch <= h:
            top = int(rng.random() * (h - ch + 1))
            left = int(rng.random() * (w - cw + 1))
            return crop_resize(x, top, left, ch, cw)
    return x


def rand_elastic(x, rng, alpha=80.0, sigma=6.0):
    # Поле нормируется по максимуму и сильно размывается, поэтому смещения малы:
    # при alpha=30 в среднем 0.4 px (снимок не меняется), при 80 — ~1.2 px, у 1 % пикселей ~4 px.
    # Начиная с ~120 края черепа становятся волнистыми.
    return elastic(x, *elastic_field(x.shape, alpha, sigma, rng))


def rand_sharpen(x, rng):
    return sharpen(x, alpha=rng.uniform(0.05, 0.15), lightness=rng.uniform(0.8, 1.0))


def aug_A(x, rng):
    """Геометрия + яркость."""
    if rng.random() < 0.5:
        x = hflip(x)
    if rng.random() < 0.8:
        x = rand_affine(x, rng, rotate=7, shift=0.025, scale=0.03)
    if rng.random() < 0.7:
        x = brightness_contrast(x, brightness=rng.uniform(-0.03, 0.03), contrast=rng.uniform(-0.1, 0.1))
    if rng.random() < 0.5:
        x = gamma(x, rng.uniform(0.9, 1.1))
    if rng.random() < 0.2:
        x = gaussian_blur(x, sigma=rng.uniform(0.5, 1.25), ksize=_blur_ksize(rng))
    return x


def aug_B(x, rng):
    """Шум и качество: разрушают подсказки по формату/разрешению файла."""
    if rng.random() < 0.3:
        x = gauss_noise(x, rng.uniform(0.005, 0.02), rng)
    if rng.random() < 0.5:
        x = jpeg(x, quality=rng.integers(70, 96))
    if rng.random() < 0.3:
        x = downscale(x, rng.uniform(0.9, 0.95))
    return x


R_POOL = [
    ("rotate", lambda x, rng: rand_affine(x, rng, rotate=7)),
    ("shift", lambda x, rng: rand_affine(x, rng, shift=0.025)),
    ("scale", lambda x, rng: rand_affine(x, rng, scale=0.03)),
    ("shear", lambda x, rng: rand_affine(x, rng, shear=3)),
    ("crop", rand_crop),
    ("elastic", rand_elastic),
    ("bright/contr", lambda x, rng: brightness_contrast(x, rng.uniform(-0.03, 0.03), rng.uniform(-0.1, 0.1))),
    ("gamma", lambda x, rng: gamma(x, rng.uniform(0.85, 1.15))),
    ("clahe", lambda x, rng: clahe(x, rng.uniform(1, 2))),
    ("sharpen", rand_sharpen),
    ("blur", lambda x, rng: gaussian_blur(x, rng.uniform(0.5, 1.25), _blur_ksize(rng))),
    ("noise", lambda x, rng: gauss_noise(x, rng.uniform(0.005, 0.02), rng)),
    ("jpeg", lambda x, rng: jpeg(x, rng.integers(65, 96))),
    ("downscale", lambda x, rng: downscale(x, rng.uniform(0.85, 0.95))),
]


def aug_R(x, rng):
    """RandAugment-подобная политика: отражение + 2 разные случайные операции из R_POOL (14 штук)."""
    if rng.random() < 0.5:
        x = hflip(x)
    for i in sorted(rng.choice(len(R_POOL), size=2, replace=False)):  # в порядке пула, как в albumentations
        x = R_POOL[i][1](x, rng)
    return x


# Уровень -> политики, которые применяются по очереди.
# В AB сначала геометрия и яркость (A), потом «качество файла» (B: шум, JPEG, даунскейл).
POLICIES = {
    "none": (),
    "A": (aug_A,),
    "B": (aug_B,),
    "AB": (aug_A, aug_B),
    "R": (aug_R,),
}
LEVELS = tuple(POLICIES)


def augment(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """
    img — uint8 224x224 после prepare(); level — один из LEVELS ("none", "A", "B", "AB", "R");
    rng — np.random.default_rng(seed). Возвращает новый uint8 224x224.
    """
    if level not in POLICIES:
        raise ValueError(f"level: один из {LEVELS}, получено {level!r}")
    x = to_float(img)
    for policy in POLICIES[level]:
        x = policy(x, rng)
    return to_uint8(x)
