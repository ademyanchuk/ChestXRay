import torch
from torch import nn


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