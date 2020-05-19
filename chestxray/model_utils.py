"""Related with NN models"""
import numpy as np
import torch
import torch.nn as nn

from chestxray.config import CFG


# Number of trainable parameters
def trainable_params(model):
    result = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has: {result} trainable parameters")


# Loss @ Init
def cce_loss_at_init(
    model, num_classes, inp_shape=(64, 3, 256, 256), criterion=nn.CrossEntropyLoss()
):
    input = torch.randn(*inp_shape, requires_grad=True)
    target = torch.empty(inp_shape[0], dtype=torch.long).random_(num_classes)
    model.eval()
    output = model(input)
    loss = criterion(output, target)
    print(
        f"""
        CCE loss @ init: {loss} -- -log(1/{num_classes} classes):
        {-np.log(1/num_classes)}"""
    )


# Init bias last Layer
def init_last_layer_bias(model, cls_probas):
    # for Categorical Crossetnropy loss
    assert CFG.target_size == len(cls_probas)
    last_layer_bias = np.log(cls_probas)
    list(model.modules())[-1].bias.data = torch.tensor(
        last_layer_bias, dtype=list(model.modules())[-1].bias.data.dtype
    )
    return model


# initialize bias of the final layer to represent class probas we have in data
# check if it works
def check_final_linear_bias_init(model, cls_probas, inp_shape=(64, 3, 256, 256)):
    input = torch.randn(*inp_shape, requires_grad=False)
    model = init_last_layer_bias(model, cls_probas)
    model.eval()
    output = model(input)
    print(f"Class probabilities:\n {cls_probas}")
    print(
        f"Softmax on output logits:\n {nn.functional.softmax(output, dim=1).data[:5]}"
    )
