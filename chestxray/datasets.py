"""Data dispatching and processing related module"""
import functools
import operator

import albumentations as A
import cv2
import numpy as np
import skimage.io
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from chestxray.config import CFG
from chestxray.config import PANDA_IMGS
from chestxray.config import TILES_IMGS


augs_dict = {
    "heavy": A.Compose(
        [
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.11,
                        shift_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    ),
                    A.NoOp(),
                ]
            ),
            A.RandomSizedCrop(
                min_max_height=(int(CFG.img_height * 0.75), CFG.img_height),
                height=CFG.img_height,
                width=CFG.img_width,
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3
                    ),
                    A.RandomGamma(gamma_limit=(50, 150)),
                    A.NoOp(),
                ]
            ),
            A.OneOf(
                [
                    A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
                    A.NoOp(),
                ]
            ),
            A.OneOf([A.CLAHE(), A.NoOp()]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # This transformation first / 255. -> scale to [0,1] and
            # then - mean and / by std
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            # Convert to torch tensor and swap axis to make Chanel first
            ToTensorV2(),
        ]
    ),
    "light": A.Compose(
        [
            A.Flip(),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3
            ),
            # This transformation first / 255. -> scale to [0,1] and
            # then - mean and / by std
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            # Convert to torch tensor and swap axis to make Chanel first
            ToTensorV2(),
        ]
    ),
}

no_aug = A.Compose(
    [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2()]
)


def get_transforms(*, data, aug="light"):
    """Choose mode `train` or `valid` and aug type:
    `light` or `heavy`"""

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
    tiles = make_tiles(img, **kwargs)
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
        image = img_to_tiles(
            image,
            num_tiles=CFG.num_tiles,
            is_train=self.is_train,
            tile_h=CFG.tile_sz,
            tile_w=CFG.tile_sz,
        )
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
    def __init__(self, df, is_train=True, transform=None):
        self.df = df
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        file_path = (
            f"../input/prostate-cancer-grade-assessment/test_images/{file_id}.tiff"
        )
        image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        image = img_to_tiles(
            image,
            num_tiles=CFG.num_tiles,
            is_train=self.is_train,
            tile_h=CFG.tile_sz,
            tile_w=CFG.tile_sz,
        )
        image = cv2.resize(
            image, (CFG.img_height, CFG.img_width), interpolation=cv2.INTER_AREA
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image


# Patch Version of Dataset from https://www.kaggle.com/hengck23/kernel16867b0575
def make_patch(image, patch_size, num_patch):
    h, w = image.shape[:2]
    s = patch_size

    pad_x = int(patch_size * np.ceil(w / patch_size) - w)
    pad_y = int(patch_size * np.ceil(h / patch_size) - h)
    image = cv2.copyMakeBorder(
        image, 0, pad_y, 0, pad_x, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    h, w = image.shape[:2]

    patch = image.reshape(h // s, s, w // s, s, 3)
    patch = patch.transpose(0, 2, 1, 3, 4).reshape(-1, s, s, 3)

    n = len(patch)
    index = np.argsort(patch.reshape(n, -1).sum(-1))[:num_patch]

    y = s * (index // (w // s))
    x = s * (index % (w // s))
    coord = np.stack([x, y, x + s, y + s]).T

    patch = patch[index]
    if len(patch) < num_patch:
        n = num_patch - len(patch)
        patch = np.concatenate(
            [patch, np.full((n, patch_size, patch_size, 3), 255, dtype=np.uint8)], 0
        )
        coord = np.concatenate([coord, np.full((n, 4), -1)], 0)
    return patch, coord


class PatchTrainDataset(Dataset):
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

        patch, coord = make_patch(image, patch_size=256, num_patch=12)
        patch = patch.astype(np.float32) / 255
        patch = patch.transpose(0, 3, 1, 2)
        patch = np.ascontiguousarray(patch)

        #         if self.transform:
        #             augmented = self.transform(image=image)
        #             image = augmented["image"]

        label = self.labels[idx]

        item = (patch, label)
        if self.debug:
            item = (patch, label, file_id)

        return item
