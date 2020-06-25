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
    num_tiles = 64
    tile_sz = 168
    batch_size = 8
    accum_step = 1  # effective batch size will be batch_size * accum_step
    dataset = "patch"  # "patch", "tiles", "lazy", "hdf5"
    aux_tile = False  # for Tiles Dataset
    aux_tile_sz = 0  # squares produced from both tile sizes need to be same size
    aux_tile_num = 0  # see above
    aug_type = "light"  # "light" or "heavy"
    # model
    att = True  # use attention for MIL-pooling, only for patch
    arch = "resnet34"  # "resnet34", "resnet50", "bitM", "efnet"
    enet_bone = "efficientnet-b0"
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    pre_init_fc_bias = False
    # loss
    ohem = False  # will work with ohem and bce
    loss = "bce"  # "cce" or "ls_soft_ce", "ohem", "bce"
    # optim
    optim = "radam"  # "adam", "sgd" or "radam"
    lr = 1e-3 if optim == "sgd" else 3e-4
    # schedule
    schedule_type = "one_cycle"  # "one_cycle", "reduce_on_plateau" or "cawr"
    oc_final_div_factor = 1e1
    cawr_T = 1
    cawr_Tmult = 2
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    resume = False
    prev_exp = "None"
    from_epoch = 0
    stage = 0
    epoch = 45
    n_fold = 4
    use_amp = True
    # Experiment
    descript = "bce + rn34 + one cycle + 168x64 patch + att"


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


def stack_sorted(tiles, ids):
    num_tiles = len(tiles)
    step = np.sqrt(num_tiles).astype(int)
    stacked = np.vstack(
        [np.hstack(tiles[ids[i : i + step]]) for i in range(0, num_tiles, step)]
    )
    return stacked


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


class TilesModel(nn.Module):
    def __init__(
        self, arch="resnet50", n=CFG.target_size, pretrained=True, loss=CFG.loss
    ):
        super().__init__()
        assert arch in ["resnet50", "resnet34"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }

        self.loss = loss
        # if we use BCE loss, need n-1 outputs
        if self.loss in ["bce"]:
            n -= 1

        model = model_dict[arch](pretrained=pretrained)

        self.encoder = nn.Sequential(*list(model.children())[:-2])
        num_ftrs = list(model.children())[-1].in_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
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
        batch_size = x.shape[0]
        x = self.encoder(x)  # x -> bs x C(Maps) x H(Maps) x W(Maps)
        avg_x = self.avgpool(x)
        max_x = self.maxpool(x)
        x = torch.cat([avg_x, max_x], dim=1).view(batch_size, -1)
        x = self.head(x)
        return x


class GatedAttention(nn.Module):
    def __init__(self, f_in):
        super(GatedAttention, self).__init__()
        self.f_in = f_in
        self.h = 128

        self.attention_V = nn.Sequential(nn.Linear(self.f_in, self.h), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.f_in, self.h), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.h, 1)

    def forward(self, x):
        # x has shape: b_sz x inst_sz x f_in_sz
        a_v = self.attention_V(x)  # b_sz x inst_sz x h
        a_u = self.attention_U(x)  # b_sz x inst_sz x h
        a = self.attention_weights(a_v * a_u)  # b_sz x inst_sz x 1
        a = torch.transpose(a, 2, 1)  # b_sz x 1 x inst_sz
        a = F.softmax(a, dim=2)  # softmax over inst_sz
        return a


class AttentionModel(nn.Module):
    def __init__(
        self, arch="resnet50", n=CFG.target_size, pretrained=True, loss=CFG.loss,
    ):
        super().__init__()
        assert arch in ["resnet50", "resnet34"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }

        self.loss = loss
        # if we use BCE loss, need n-1 outputs
        if self.loss in ["bce"]:
            n -= 1

        # create back bone
        model = model_dict[arch](pretrained=pretrained)
        # define feature encoder
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        num_ftrs = list(model.children())[-1].in_features
        # define gated attention module
        self.attention = GatedAttention(num_ftrs)
        # define classifier
        if CFG.model_cls == "deep":
            self.head = nn.Sequential(
                OrderedDict(
                    [
                        ("cls_fc", nn.Linear(num_ftrs, 512)),
                        ("cls_bn", nn.BatchNorm1d(512)),
                        ("cls_relu", nn.ReLU(inplace=True)),
                        ("cls_logit", nn.Linear(512, n)),
                    ]
                )
            )
        elif CFG.model_cls == "one_layer":
            self.head = nn.Sequential(
                OrderedDict([("cls_logit", nn.Linear(num_ftrs, n))])
            )

        del model

    def forward(self, x):
        batch_size, num_patch, C, H, W = x.shape

        x = x.view(-1, C, H, W)  # x -> bs*num_patch x C x H x W
        # extract features
        x = self.encoder(x)  # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)
        # reduce dimensionality of the feature vector
        x = F.adaptive_avg_pool2d(x, (1, 1))  # x -> bs*num_patch x C(Maps) x 1 x 1
        x = x.view(batch_size, num_patch, -1)
        # x -> bs x num_patch x C (C is now feature size)

        # get attention
        att = self.attention(x)  # att -> bs x 1 x num_patch
        # MIL attended pooling to get Bag-Embeeding from Instance Embeedings
        x = torch.matmul(att, x)  # x -> bs x 1 x C
        x = x.view(batch_size, -1)  # x -> bs x C
        # Classification on the Bag-Embeeding Level
        x = self.head(x)  # x -> bs x n
        return x  # only logits
