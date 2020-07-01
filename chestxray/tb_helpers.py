import numpy as np
import torchvision

from chestxray.train_utils import compute_preds
from chestxray.visualize import plot_confusion_matrix
from chestxray.visualize import reverse_show_img


# write to TensorBoard helpers
def weights_to_tb(writer, model, step=0):
    conv1_weight = list(model.parameters())[0].data.to("cpu")
    img_grid = torchvision.utils.make_grid(conv1_weight.float(), normalize=True)
    writer.add_image(tag="Model conv1 weights", img_tensor=img_grid, global_step=step)


def input_img_to_tb(writer, inputs, step):
    img = reverse_show_img(inputs[0], denorm=True)
    writer.add_image(
        tag="Input Image", img_tensor=img, global_step=step, dataformats="HWC",
    )
    del img


def text_preds_to_tb(writer, outputs, labels, step, **kwargs):
    preds, targs = compute_preds(outputs, labels, **kwargs)
    preds_text = f"targs: {targs}, preds: {preds}"
    writer.add_text("Actuals vs Predictions", preds_text, global_step=step)


def metrics_to_tb(writer, mode, train_loss, train_score, val_loss, val_score, step):
    writer.add_text(
        f"On best {mode} save:",
        f"tr_loss: {train_loss:.4f}, tr_score: {train_score:.4f}, val_loss: {val_loss:.4f}, val_score: {val_score:.4f}",  # noqa
        global_step=step,
    )


def conf_matrix_to_tb(writer, val_epoch_labels, val_epoch_preds, step):
    writer.add_figure(
        tag="Confusion matrix",
        figure=plot_confusion_matrix(val_epoch_labels, val_epoch_preds, normalize=True),
        global_step=step,
    )


def attention_to_tb(writer, att, step):
    att_arr = att.data.cpu().numpy()[0, 0]
    text = f"{np.around(att_arr, decimals=3)}"
    writer.add_text("Attention Map for Oth Val sample", text, global_step=step)
