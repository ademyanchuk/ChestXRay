from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
    def __init__(self, arch="resnet50", n=6, pretrained=True):
        super().__init__()
        assert arch in ["resnet50", "resnet34"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }

        model = model_dict[arch](pretrained=pretrained)

        self.encoder = nn.Sequential(*list(model.children())[:-2])
        num_ftrs = list(model.children())[-1].in_features
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
        del model

    def forward(self, x):
        batch_size, num_patch, C, H, W = x.shape

        x = x.view(-1, C, H, W)  # x -> bs*num_patch x C x H x W
        x = self.encoder(x)  # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)

        x = aggregate(x, batch_size, num_patch)
        x = self.head(x)
        return x
