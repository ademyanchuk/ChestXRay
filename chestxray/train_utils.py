def compute_preds(outputs, labels, task_type):
    if task_type in ["bce", "focal"]:  # here is bce version of focal loss
        preds = outputs.detach().sigmoid().sum(1)
        if labels is not None:
            targs = labels.sum(1)
    else:
        preds = outputs.detach().sigmoid()
        if labels is not None:
            targs = labels
    if labels is not None:
        return preds, targs
    return preds
