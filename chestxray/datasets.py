"""Data dispatching and processing related module"""
import functools
import operator
from pathlib import Path

import albumentations as A
import cv2
import h5py
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from chestxray.config import CFG
from chestxray.config import PANDA_IMGS
from chestxray.config import TILES_IMGS

TEST_PATH = Path("../input/prostate-cancer-grade-assessment/test_images")

TV_MEAN = [0.485, 0.456, 0.406]
TV_STD = [0.229, 0.224, 0.225]

BIT_MEAN = [0.5, 0.5, 0.5]
BIT_STD = [0.5, 0.5, 0.5]

augs_dict = {
    "heavy": A.Compose(
        [
            A.Flip(),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ]
    ),
    "light": A.Compose(
        [
            A.Flip(),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            # This transformation first / 255. -> scale to [0,1] and
            # then - mean and / by std
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            # # Convert to torch tensor and swap axis to make Chanel first
            # ToTensorV2(),
        ]
    ),
}

normalize = A.Normalize(mean=TV_MEAN, std=TV_STD)


def get_transforms(*, data, aug="light"):
    """Choose mode `train` or `valid` and aug type:
    `light` or `heavy`"""

    assert data in ("train", "valid")
    assert aug in ("light", "heavy")

    if data == "train":
        return augs_dict[aug]

    elif data == "valid":
        return None


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


def stack_sorted(tiles, ids):
    num_tiles = len(tiles)
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


class TilesTrainDataset(Dataset):
    def __init__(
        self,
        df,
        is_train=True,
        transform=None,
        suffix="tiff",
        debug=CFG.debug,
        loss=CFG.loss,
        aux_tile=CFG.aux_tile,
        regression=False,
    ):
        self.df = df
        self.labels = df[CFG.target_col].values
        self.transform = transform
        self.is_train = is_train
        self.suffix = suffix
        self.debug = debug
        self.loss = loss
        self.aux_tile = aux_tile
        self.regression = regression

    def _make_image(self, image, num_tiles, tile_sz):
        # Make sure we can do square
        assert int(np.sqrt(num_tiles)) == np.sqrt(num_tiles)
        patch, _ = make_patch(image, patch_size=tile_sz, num_patch=num_tiles)

        if self.is_train:
            ids = np.random.choice(range(len(patch)), size=len(patch), replace=False)
        else:
            ids = np.arange(len(patch))
        image = stack_sorted(patch, ids)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        normalized = normalize(image=image)
        image = normalized["image"]

        image = image.transpose(2, 0, 1)  # to Chanel first
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        if self.suffix == "tiff":
            file_path = f"{PANDA_IMGS}/{file_id}.{self.suffix}"
            image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        elif self.suffix == "jpeg":
            file_path = f"{PANDA_IMGS}/{file_id}_1.{self.suffix}"
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.aux_tile:
            image = self._make_image(image, CFG.num_tiles, CFG.tile_sz)

        else:
            if self.is_train:
                use_aux = np.random.choice([True, False])
                if use_aux:  # use auxillary tile size and number
                    image = self._make_image(image, CFG.aux_tile_num, CFG.aux_tile_sz)
                else:
                    image = self._make_image(image, CFG.num_tiles, CFG.tile_sz)
            else:
                image_main = self._make_image(image, CFG.num_tiles, CFG.tile_sz)
                image_aux = self._make_image(image, CFG.aux_tile_num, CFG.aux_tile_sz)
                image = (image_main, image_aux)

        # if use bce, make label as bit encoded vector
        if self.loss == "bce":
            label = np.zeros(CFG.target_size - 1).astype(np.float32)
            label[: self.labels[idx]] = 1.0
        elif self.regression:
            label = self.labels[idx].astype(np.float32)
        else:
            label = self.labels[idx]

        item = (image, label)  # if using aux tile, image will be tuple!!!
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
    def __init__(
        self,
        df,
        transform=None,
        suffix="tiff",
        img_path=TEST_PATH,
        aux_tile=CFG.aux_tile,
    ):
        self.df = df
        self.transform = transform
        self.suffix = suffix
        self.img_path = img_path
        self.aux_tile = aux_tile

    def _make_image(self, image, num_tiles, tile_sz):
        # Make sure we can do square
        assert int(np.sqrt(num_tiles)) == np.sqrt(num_tiles)
        patch, _ = make_patch(image, patch_size=tile_sz, num_patch=num_tiles)

        ids = np.arange(len(patch))
        image = stack_sorted(patch, ids)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        normalized = normalize(image=image)
        image = normalized["image"]

        image = image.transpose(2, 0, 1)  # to Chanel first
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]

        if self.suffix == "tiff":
            file_path = f"{self.img_path}/{file_id}.{self.suffix}"
            image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        elif self.suffix == "jpeg":
            file_path = f"{self.img_path}/{file_id}_1.{self.suffix}"
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.aux_tile:
            image = self._make_image(image, CFG.num_tiles, CFG.tile_sz)

        else:
            image_main = self._make_image(image, CFG.num_tiles, CFG.tile_sz)
            image_aux = self._make_image(image, CFG.aux_tile_num, CFG.aux_tile_sz)
            image = (image_main, image_aux)

        return image


class PatchTrainDataset(Dataset):
    def __init__(
        self,
        df,
        is_train=True,
        transform=None,
        suffix="tiff",
        debug=CFG.debug,
        loss=CFG.loss,
    ):
        self.df = df
        self.labels = df[CFG.target_col].values
        self.is_train = is_train
        self.transform = transform
        self.suffix = suffix
        self.debug = debug
        self.loss = loss

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        if self.suffix == "tiff":
            file_path = f"{PANDA_IMGS}/{file_id}.{self.suffix}"
            image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        elif self.suffix == "jpeg":
            file_path = f"{PANDA_IMGS}/{file_id}_1.{self.suffix}"
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        patch, coord = make_patch(
            image, patch_size=CFG.tile_sz, num_patch=CFG.num_tiles
        )

        if self.is_train:
            ids = np.random.choice(range(len(patch)), size=len(patch), replace=False)
            patch = patch[ids]
        # augment sequence
        if self.transform:
            for i in range(len(patch)):
                augmented = self.transform(image=patch[i])
                patch[i] = augmented["image"]

        if CFG.arch != "bitM":
            normalize = A.Normalize(mean=TV_MEAN, std=TV_STD)
        else:
            normalize = A.Normalize(mean=BIT_MEAN, std=BIT_STD)

        normalized = normalize(image=patch)
        patch = normalized["image"]

        patch = patch.transpose(0, 3, 1, 2)
        patch = np.ascontiguousarray(patch)

        # if use bce, make label as bit encoded vector
        if self.loss == "bce":
            label = np.zeros(CFG.target_size - 1).astype(np.float32)
            label[: self.labels[idx]] = 1.0
        else:
            label = self.labels[idx]

        item = (patch, label)
        if self.debug:
            item = (patch, label, file_id)

        return item


class PatchTestDataset(Dataset):
    def __init__(self, df, transform=None, img_path=TEST_PATH, suffix="tiff"):
        self.df = df
        self.transform = transform
        self.img_path = img_path
        self.suffix = suffix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df[CFG.img_id_col].values[idx]
        if self.suffix == "tiff":
            file_path = f"{self.img_path}/{file_id}.{self.suffix}"
            image = skimage.io.MultiImage(file_path)[CFG.tiff_layer]
        elif self.suffix == "jpeg":
            file_path = f"{self.img_path}/{file_id}_1.{self.suffix}"
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        patch, coord = make_patch(
            image, patch_size=CFG.tile_sz, num_patch=CFG.num_tiles
        )
        # augment sequence
        if self.transform:
            for i in range(len(patch)):
                augmented = self.transform(image=patch[i])
                patch[i] = augmented["image"]

        normalized = normalize(image=patch)
        patch = normalized["image"]

        patch = patch.transpose(0, 3, 1, 2)
        patch = np.ascontiguousarray(patch)

        return patch


class H5PatchDataset(Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset
        (one or multiple HDF5 files).
        fnames: h5 files to use.
        data_cache_size: Number of HDF5 files that can be cached in
        the cache (default=2).
    """

    def __init__(self, file_path, fnames, data_cache_size=2):
        super().__init__()
        self.data_cache_size = data_cache_size

        # Collect all h5 file paths
        p = Path(file_path)
        assert p.is_dir()
        h5files = [p / fname for fname in fnames]
        if len(h5files) < 1:
            raise RuntimeError("No hdf5 datasets found")
        self.h5files = h5files

        self._full_len, self._common_len = self._get_lengths()
        self._cache = {}  # key - file idx in self.h5files, value - opened file

    def _get_lengths(self):
        lenghts = []
        for h5file in self.h5files:
            with h5py.File(f"{h5file}", "r") as file:
                lenghts.append(len(file["/images"]))
        return sum(lenghts), max(lenghts)

    def _load_data(self, file_idx):
        file = h5py.File(f"{self.h5files[file_idx]}", "r")
        self._cache[file_idx] = file
        # print(f"Add file {file_idx} to cache")

        if len(self._cache) > self.data_cache_size:
            # remove min as the sampler randomly took indexes from bins
            # i.e. from (0:100), (100:200) and so on in sequence
            remove_idx = min(self._cache)
            self._cache[remove_idx].close()
            self._cache.pop(remove_idx)

    def _get_data(self, file_idx, effective_idx, group):
        data = self._cache[file_idx][f"/{group}"][effective_idx]
        return data

    def __getitem__(self, index):
        file_idx = index // self._common_len
        if file_idx not in self._cache:
            self._load_data(file_idx)

        effective_idx = index - (file_idx * self._common_len)

        patch = self._get_data(file_idx, effective_idx, "images")
        patch = patch.astype(np.uint8)
        normalized = normalize(image=patch)  # from albumentations
        patch = normalized["image"]

        patch = patch.transpose(0, 3, 1, 2)  # to num_patch x C x H x W
        patch = np.ascontiguousarray(patch)

        label = self._get_data(file_idx, effective_idx, "labels")
        label = label.astype(np.int)
        return patch, label

    def __len__(self):
        return self._full_len


class SeqenceRandomSampler(Sampler):
    def __init__(self, full_len, step):
        self.full_len = full_len
        self.step = step

    def __iter__(self):
        result = []
        indices = list(range(self.full_len))
        for i in range(0, self.full_len, self.step):
            part = indices[i : i + self.step]
            np.random.shuffle(part)
            result.extend(part)
        return (i for i in result)

    def __len__(self):
        return self.full_len
