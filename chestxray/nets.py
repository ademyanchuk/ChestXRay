"""Model architectures definitions"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from chestxray.config import CFG


# Convolutional neural network (two convolutional layers) - tiny ConvNet
class TinyConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# Convolutional neural network (two convolutional layers) - tiny ConvNet
class TinyV2ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyV2ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def make_RN50_cls(pretrained=True):
    model_ft = models.resnet50(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    if CFG.model_cls == "deep":
        model_ft.fc = nn.Sequential(
            OrderedDict(
                [
                    ("cls_lin1", nn.Linear(num_ftrs, 512)),
                    ("cls_relu", nn.ReLU(inplace=True)),
                    ("cls_bn", nn.BatchNorm1d(512)),
                    ("drop", nn.Dropout(0.5)),
                    ("cls_lin2", nn.Linear(512, CFG.target_size)),
                ]
            )
        )
    elif CFG.model_cls == "one_layer":
        model_ft.fc = nn.Linear(num_ftrs, CFG.target_size)
    return model_ft


def freeze_botom(model):
    for name, group in model.named_children():
        if name in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]:
            print(f"Freezing layer {name}..")
            for p in group.parameters():
                p.requires_grad = False


def make_RN18_cls(pretrained=True):
    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, CFG.target_size)
    return model_ft


# Model to work with Patches
def aggregate(x, batch_size, num_patch):
    _, C, H, W = x.shape
    x = x.view(batch_size, num_patch, C, H, W)
    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, C, num_patch * H, W)

    avg = F.adaptive_avg_pool2d(x, 1)
    max_ = F.adaptive_max_pool2d(x, 1)
    x = torch.cat([avg, max_], 1).view(batch_size, -1)
    return x


class TilesModel(nn.Module):
    def __init__(self, arch="resnet50", n=CFG.target_size, pretrained=True):
        super().__init__()
        assert arch in ["resnet50", "resnet34"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }

        self.model = model_dict[arch](pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        if CFG.model_cls == "deep":
            self.model.fc = nn.Sequential(
                OrderedDict(
                    [
                        ("cls_lin1", nn.Linear(num_ftrs, 512)),
                        ("cls_relu", nn.ReLU(inplace=True)),
                        ("cls_bn", nn.BatchNorm1d(512)),
                        ("cls_lin2", nn.Linear(512, CFG.target_size)),
                    ]
                )
            )
        elif CFG.model_cls == "one_layer":
            self.model.fc = nn.Linear(num_ftrs, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


class PatchModel(nn.Module):
    def __init__(self, arch="resnet50", n=CFG.target_size, pretrained=True):
        super().__init__()
        assert arch in ["resnet50", "resnet34"]
        model_dict = {
            "resnet50": models.resnet50,
            "resnet34": models.resnet34,
        }

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
                        ("cls_logit", nn.Linear(512, CFG.target_size)),
                    ]
                )
            )
        elif CFG.model_cls == "one_layer":
            self.head = nn.Sequential(
                OrderedDict(
                    [
                        ("cls_logit", nn.Linear(2 * num_ftrs, CFG.target_size)),
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
