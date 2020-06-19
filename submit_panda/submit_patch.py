from collections import OrderedDict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models

PANDA_PATH = Path("../input/prostate-cancer-grade-assessment")
TEST_PATH = Path("../input/prostate-cancer-grade-assessment/test_images")


class CFG:
    # overall
    debug = False
    seed = 1982
    # data
    img_height = 224
    img_width = 224
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 16
    tile_sz = 224
    batch_size = 16
    accum_step = 1  # effective batch size will be batch_size * accum_step
    dataset = "patch"  # "patch", "tiles", "lazy", "hdf5"
    multi_lvl = False  # for Patch Dataset
    aug_type = "light"  # "light" or "heavy"
    # model
    arch = "resnet34"  # "resnet34", "resnet50", "bitM", "efnet"
    enet_bone = "efficientnet-b0"
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    pre_init_fc_bias = True
    # loss
    loss = "bce"  # "cce" or "ls_soft_ce", "ohem", "bce"
    # optim
    optim = "radam"  # "adam", "sgd" or "radam"
    lr = 1e-3 if optim == "sgd" else 3e-4
    # schedule
    schedule_type = "one_cycle"  # "one_cycle", "reduce_on_plateau" or "cawr"
    oc_final_div_factor = 1e2
    cawr_T = 1
    cawr_Tmult = 2
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    resume = False
    prev_exp = "None"
    from_epoch = 0
    stage = 0
    epoch = 35
    n_fold = 4
    use_amp = True
    # Experiment
    descript = "bce + rn34 + one cycle + 224x16"


# Datasets

normalize = A.Compose(
    [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)]
)


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


# Model to work with Patches
def aggregate(x, batch_size, num_patch):
    _, C, H, W = x.shape
    x = x.view(batch_size, num_patch, C, H, W)
    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, C, num_patch * H, W)

    avg = F.adaptive_avg_pool2d(x, 1)
    max_ = F.adaptive_max_pool2d(x, 1)
    x = torch.cat([avg, max_], 1).view(batch_size, -1)
    return x


class PatchModel(nn.Module):
    def __init__(self, arch="resnet50", n=CFG.target_size, pretrained=True):
        super().__init__()
        assert arch in ["resnet50", "resnet34", "bitM"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }
        # if we use BCE loss, need n-1 outputs
        if CFG.loss == "bce":
            n -= 1

        if arch.startswith("resnet"):
            model = model_dict[arch](pretrained=pretrained)

            self.encoder = nn.Sequential(*list(model.children())[:-2])
            num_ftrs = list(model.children())[-1].in_features
            if CFG.model_cls == "deep":
                self.head = nn.Sequential(
                    OrderedDict(
                        [
                            ("cls_fc", nn.Linear(2 * num_ftrs, 512)),
                            ("cls_bn", nn.BatchNorm1d(512)),
                            ("cls_relu", nn.ReLU(inplace=True)),
                            ("cls_logit", nn.Linear(512, n)),
                        ]
                    )
                )
            elif CFG.model_cls == "one_layer":
                self.head = nn.Sequential(
                    OrderedDict([("cls_logit", nn.Linear(2 * num_ftrs, n))])
                )

        del model

    def forward(self, x):
        batch_size, num_patch, C, H, W = x.shape

        x = x.view(-1, C, H, W)  # x -> bs*num_patch x C x H x W
        x = self.encoder(x)  # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)

        x = aggregate(x, batch_size, num_patch)
        x = self.head(x)
        return x
