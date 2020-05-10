"""Data dispatching and processing related module"""
import functools
import operator

import cv2
import numpy as np
import skimage.io
import torch
from albumentations import Compose
from albumentations import Flip
from albumentations import GaussNoise
from albumentations import HueSaturationValue
from albumentations import Normalize
from albumentations import RandomBrightnessContrast
from albumentations import ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from chestxray.config import CFG
from chestxray.config import PANDA_IMGS
from chestxray.config import TILES_IMGS


augs_dict = {
    "heavy": Compose(
        [
            Flip(),
            GaussNoise(),
            RandomBrightnessContrast(),
            HueSaturationValue(),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3
            ),
            # This transformation first / 255. -> scale to [0,1] and
            # then - mean and / by std
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            # Convert to torch tensor and swap axis to make Chanel first
            ToTensorV2(),
        ]
    ),
    "light": Compose(
        [
            Flip(),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3
            ),
            # This transformation first / 255. -> scale to [0,1] and
            # then - mean and / by std
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            # Convert to torch tensor and swap axis to make Chanel first
            ToTensorV2(),
        ]
    ),
}

no_aug = Compose(
    [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2()]
)


def get_transforms(*, data, aug="light"):

    assert data in ("train", "valid")
    assert aug in ("light", "heavy")

    if data == "train":
        return augs_dict[aug]

    elif data == "valid":
        return no_aug


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
        image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        item = (image, label)
        if self.debug:
            item = (image, label, file_id)

        return item


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = (
            f"../input/prostate-cancer-grade-assessment/test_images/{file_id}.tiff"
        )
        image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image


# Make iles from image and stack thresholding by white-ish pixels


def make_tiles(img, tile_h=128, tile_w=128):
    img_h, img_w = img.shape[0], img.shape[1]
    steps_h = np.floor(img_h / tile_h).astype(np.int)
    steps_w = np.floor(img_w / tile_w).astype(np.int)
    tiles = []
    for i in range(steps_h):
        for j in range(steps_w):
            tile = img[
                i * tile_h : i * tile_h + tile_h, j * tile_w : j * tile_w + tile_w, :
            ]
            tiles.append(tile)
    return np.array(tiles)


def pxl_percentage(img, above_thresh=230):
    # percent of white-ish pixels in img
    whitish = (img > above_thresh).sum()
    pix_num = functools.reduce(operator.mul, img.shape)
    return whitish / pix_num


def stack_sorted(tiles, ids, num_tiles):
    # lets try hard-code 6x6 blocks
    step = np.sqrt(num_tiles).astype(int)
    stacked = np.vstack(
        [np.hstack(tiles[ids[i : i + step]]) for i in range(0, num_tiles, step)]
    )
    return stacked


def get_weighted_sample_ids(white_pcts, num_tiles):
    tiles_ids = np.arange(len(white_pcts))
    white_mask = white_pcts == 1.0
    probas = np.empty_like(tiles_ids, dtype=float)
    probas.fill(1 / len(white_pcts))
    # zero fully white tiles probas
    probas[white_mask] = 1e-8
    wght_probas = probas / (white_pcts + 1e-8)
    wght_probas /= wght_probas.sum()
    ids = np.random.choice(tiles_ids, num_tiles, replace=False, p=wght_probas)
    return ids


def img_to_tiles(img, num_tiles=36, is_train=True, *args, **kwargs):
    # Put all together
    tiles = make_tiles(img)
    if len(tiles) < num_tiles:
        return img
    white_pcts = np.array([pxl_percentage(tile) for tile in tiles])
    if is_train:
        gradient_ids = get_weighted_sample_ids(white_pcts, num_tiles)
    else:
        gradient_ids = np.argsort(white_pcts)

    return stack_sorted(tiles, gradient_ids, num_tiles)


class TilesTrainDataset(Dataset):
    def __init__(self, df, is_train=True, transform=None, debug=CFG.debug):
        self.df = df
        self.labels = df[CFG.target_col].values
        self.transform = transform
        self.is_train = is_train
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = f"{PANDA_IMGS}/{file_id}.tiff"
        image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        # use stochastic tiles compose for train and deterministic for valid
        image = img_to_tiles(image, is_train=self.is_train)
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        item = (image, label)
        if self.debug:
            item = (image, label, file_id)

        return item


class LazyTilesDataset(Dataset):
    def __init__(self, df, transform=None, debug=CFG.debug):
        self.df = df
        self.labels = df[CFG.target_col].values
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = f"{TILES_IMGS}/{file_id}.png"
        image = skimage.io.imread(file_path)
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        item = (image, label)
        if self.debug:
            item = (image, label, file_id)

        return item


class TilesTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = (
            f"../input/prostate-cancer-grade-assessment/test_images/{file_id}.tiff"
        )
        image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        image = img_to_tiles(image, is_train=False)
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image
