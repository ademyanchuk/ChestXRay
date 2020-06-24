"""Model architectures definitions"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import models

from chestxray.bit_net import ResNetV2
from chestxray.bit_net import weights_from_cache
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


def aggregateBiT(x, batch_size, num_patch):
    _, C, H, W = x.shape
    x = x.view(batch_size, num_patch, C, H, W)
    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, C, num_patch * H, W)

    x = F.adaptive_avg_pool2d(x, 1)
    return x


def make_BiT_model(num_classes=CFG.target_size, pretrained=True):
    print("Loading BiT-M weights..")
    weights = weights_from_cache("BiT-M-R50x1")
    model = ResNetV2(
        ResNetV2.BLOCK_UNITS["r50"],
        width_factor=1,
        head_size=num_classes,
        zero_head=True,
    )
    if pretrained:
        model.load_from(weights)
    return model


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


class PatchModel(nn.Module):
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


class PatchEnetModel(nn.Module):
    def __init__(
        self,
        backbone="efficientnet-b0",
        n=CFG.target_size,
        pretrained=True,
        loss=CFG.loss,
    ):
        super().__init__()
        assert backbone in ["efficientnet-b0", "efficientnet-b3"]
        self.loss = loss
        # if we use BCE loss, need n-1 outputs
        if self.loss in ["bce"]:
            n -= 1

        if pretrained:
            self.model = EfficientNet.from_pretrained(backbone)
        else:
            self.model = EfficientNet.from_name(backbone)

        num_ftrs = self.model._fc.in_features
        if CFG.model_cls == "deep":
            self.model._fc = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "cls_fc",
                            nn.Linear(2 * num_ftrs, 512),
                        ),  # agregate use concat pooling, so *2
                        ("cls_bn", nn.BatchNorm1d(512)),
                        ("cls_relu", nn.ReLU(inplace=True)),
                        ("cls_logit", nn.Linear(512, n)),
                    ]
                )
            )
        elif CFG.model_cls == "one_layer":
            self.model._fc = nn.Sequential(
                OrderedDict([("cls_logit", nn.Linear(2 * num_ftrs, n))])
            )
        del self.model._avg_pooling  # use pooling in aggregate func

    def forward(self, x):
        batch_size, num_patch, C, H, W = x.shape

        x = x.view(-1, C, H, W)  # x -> bs*num_patch x C x H x W
        x = self.model.extract_features(
            x
        )  # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)

        x = aggregate(x, batch_size, num_patch)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        return x


class PatchBiTModel(nn.Module):
    def __init__(self, n=CFG.target_size, pretrained=True):
        super().__init__()
        # if we use BCE loss, need n-1 outputs
        if CFG.loss == "bce":
            n -= 1

        self.model = make_BiT_model(num_classes=n, pretrained=pretrained)
        del self.model.head.avg  # use pooling in aggregate func

    def forward(self, x):
        batch_size, num_patch, C, H, W = x.shape

        x = x.view(-1, C, H, W)
        # x -> bs*num_patch x C x H x W
        x = self.model.body(self.model.root(x))
        x = self.model.head.relu(self.model.head.gn(x))
        # x -> bs*num_patch x C(Maps) x H(Maps) x W(Maps)

        x = aggregateBiT(x, batch_size, num_patch)
        x = self.model.head.conv(x)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]


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
        return x, att
