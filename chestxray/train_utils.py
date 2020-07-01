def compute_preds(outputs, labels, task_type):
    if task_type == "bce":
        preds = outputs.detach().sigmoid().sum(1)
        targs = labels.sum(1)
    else:
        preds = outputs.detach().sigmoid()
        targs = labels
    return preds, targs
