"""Visualization related functions"""
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from chestxray.config import CFG
from chestxray.config import PANDA_IMGS


def show_from_ids(ids, df, img_path=PANDA_IMGS):
    labels = [
        df.loc[df[CFG.img_id_col] == img_id, CFG.target_col].values[0] for img_id in ids
    ]
    paths = [f"{img_path}/{img_id}.tiff" for img_id in ids]
    plt.figure(figsize=(16, 16))
    for n in range(8):
        ax = plt.subplot(2, 4, n + 1)  # noqa
        img = skimage.io.MultiImage(paths[n])[-1]
        plt.imshow(img)
        plt.title(labels[n])
        plt.axis("off")


# Batch reversed
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(16, 16))
    for n in range(8):
        ax = plt.subplot(2, 4, n + 1)  # noqa
        img = image_batch[n]
        # Reverse all preprocessing of TrainDataset
        img = reverse_show_img(img)
        plt.imshow(img)
        plt.title(label_batch[n].numpy())
        plt.axis("off")


# one img tensor reversed
def reverse_show_img(img):
    # Reverse all preprocessing
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = (img * 255).astype(np.uint8)
    return img


# just show torch tensor
def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize=(14, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper functions


def output_to_probs(output):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    # convert output probabilities to predicted class
    out_data = output.data.type(torch.float32)
    _, preds_np = torch.max(out_data, 1)
    preds = np.squeeze(preds_np)
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, out_data)]


def plot_classes_preds(output, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "output_to_probs" function.
    """
    preds, probs = output_to_probs(output)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(14, 14))
    for idx in np.arange(8):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        plt.imshow(reverse_show_img(images[idx]))
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                preds[idx], probs[idx] * 100.0, labels[idx]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
        plt.tight_layout()
    return fig

    # from sklearn docs


def plot_confusion_matrix(
    y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is None:
        classes = np.unique(y_true)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig
