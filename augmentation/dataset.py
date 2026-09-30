"""
torch Dataset с аугментацией на лету.

Вынесен из ноутбука в модуль, чтобы DataLoader(num_workers>0) работал на Windows
(дочерние процессы должны импортировать класс, а классы из ноутбука им недоступны).
"""
import torch
from torch.utils.data import Dataset

from augment import build_transforms


class MRIDataset(Dataset):
    """
    x: uint8 [N, H, W] (уже после load_image + build_transforms("none")), y: метки [N].
    expand > 1: каждый снимок встречается expand раз за эпоху, каждый раз с новой аугментацией.
    Выход: float32 [1, H, W] в диапазоне [-1, 1] и метка.
    """

    def __init__(self, x, y, level: str = "none", expand: int = 1):
        self.x, self.y = x, y
        self.tf = build_transforms(level) if level != "none" else None
        self.expand = expand if self.tf else 1

    def __len__(self):
        return len(self.x) * self.expand

    def __getitem__(self, i):
        i %= len(self.x)
        img = self.x[i] if self.tf is None else self.tf(image=self.x[i])["image"]
        t = torch.from_numpy(img).float().div_(127.5).sub_(1.0)[None]
        return t, int(self.y[i])
