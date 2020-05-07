"""Data dispatching and processing related module"""
import cv2
import numpy as np
import skimage.io
import torch
from albumentations import Compose
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from chestxray.config import CFG
from chestxray.config import PANDA_IMGS


def get_transforms(*, data):

    assert data in ("train", "valid")

    if data == "train":
        return Compose(
            [
                # This transformation first / 255. -> scale to [0,1] and
                # then - mean and / by std
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
                # Convert to torch tensor and swap axis to make Chanel first
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
                ToTensorV2(),
            ]
        )


class ZeroDataset(Dataset):
    """To check model on nonsense data"""

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        zero_img = torch.zeros(3, 256, 256, dtype=torch.float32)
        label = np.random.randint(0, CFG.target_size)  # random labels
        return zero_img, label


class TrainDataset(Dataset):
    def __init__(self, df, transform=None, debug=CFG.debug):
        self.df = df
        self.labels = df[CFG.target_col].values
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = f"{PANDA_IMGS}/{file_id}.tiff"
        image = skimage.io.MultiImage(file_path)
        image = cv2.resize(
            image[-1], (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        item = (image, label)
        if self.debug:
            item = (image, label, file_id)

        return item
