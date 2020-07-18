from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _WeightedLoss


# from https://www.kaggle.com/c/bengaliai-cv19/discussion/130811
class SmoothLabelCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    2. Calculate loss
    """

    def __init__(self, label_smoothing=0.0):
        super(SmoothLabelCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        # When label smoothing is turned on, KL-divergence is minimized
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            self.criterion = nn.NLLLoss()
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        # Map the output to (0, 1)
        scores = self.LogSoftmax(dec_outs)
        # n_class
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


# from https://github.com/CoinCheung/pytorch-loss/blob/master/label_smooth.py
class LabelSmoothSoftmaxCEV1(nn.Module):
    """
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2
    that uses derived gradients
    """

    def __init__(self, lb_smooth=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        """
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            label = (
                torch.empty_like(logits)
                .fill_(lb_neg)
                .scatter_(1, label.unsqueeze(1), lb_pos)
                .detach()
            )

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == "mean":
            loss = loss.sum() / n_valid
        if self.reduction == "sum":
            loss = loss.sum()

        return loss


# Online Hard Examples Mining loss
class TopkCrossEntropy(_WeightedLoss):
    def __init__(
        self,
        top_k=0.7,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="none",
    ):
        super(TopkCrossEntropy, self).__init__(weight, size_average, reduce, reduction)

        self.ignore_index = ignore_index
        self.top_k = top_k
        self.loss = nn.NLLLoss(
            weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, input, target, valid=False):
        loss = self.loss(F.log_softmax(input, dim=1), target)
        # print(loss)

        if self.top_k == 1 or valid:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))
            return torch.mean(valid_loss)


class TopkBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        top_k: float = 0.7,
        reduction: str = "none",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(TopkBCEWithLogitsLoss, self).__init__()
        self.top_k = top_k
        self.criterion = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, logits, labels, valid=False):
        loss = self.criterion(logits, labels).mean(dim=1)
        # print(loss)

        if self.top_k == 1 or valid or len(loss) < 2:
            return torch.mean(loss)
        else:
            hard_loss, _ = torch.topk(loss, int(self.top_k * loss.shape[0]))
            return torch.mean(hard_loss)


# from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
