"""Model architectures definitions"""
from collections import OrderedDict

import torch.nn as nn
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
class Model(nn.Module):
    def __init__(self, arch="resnet50", n=CFG.target_size, pretrained=True):
        super().__init__()
        assert arch in ["resnet50"]
        model = models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        num_ftrs = list(model.children())[-1].in_features
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("cls_avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 1))),
                    ("cls_flat", nn.Flatten()),
                    ("cls_linear", nn.Linear(num_ftrs, CFG.target_size)),
                ]
            )
        )

    def forward(self, x):
        shape = x.shape
        num_patch = len(x[0])
        x = x.view(-1, shape[2], shape[3], shape[4])  # x -> bs*num_patch x C x H x W
        x = self.encoder(x)  # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)
        shape = x.shape
        # concatenate the output for tiles into a single map
        x = (
            x.view(-1, num_patch, shape[1], shape[2], shape[3])
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, shape[1], shape[2] * num_patch, shape[3])
        )
        x = self.head(x)
        return x
