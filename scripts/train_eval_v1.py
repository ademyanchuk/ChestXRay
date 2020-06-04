#!/usr/bin/env python
# coding: utf-8
import subprocess
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from apex import amp
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chestxray.config import CFG
from chestxray.config import MODEL_PATH
from chestxray.config import PANDA_PATH
from chestxray.config import TRAIN_CSV
from chestxray.datasets import get_transforms
from chestxray.datasets import H5PatchDataset
from chestxray.datasets import LazyTilesDataset
from chestxray.datasets import PatchTrainDataset
from chestxray.datasets import SeqenceRandomSampler
from chestxray.datasets import TilesTrainDataset
from chestxray.losses import LabelSmoothSoftmaxCEV1
from chestxray.misc import seed_torch
from chestxray.model_utils import init_last_layer_bias
from chestxray.nets import freeze_botom
from chestxray.nets import PatchModel
from chestxray.optimizers import RAdam
from chestxray.visualize import plot_classes_preds
from chestxray.visualize import plot_confusion_matrix
from chestxray.visualize import reverse_show_img
from chestxray.visualize import text_classes_preds


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_properties(device))


# Fix Random Seed
seed_torch()


# Load Data
TRAIN_DF = pd.read_csv(TRAIN_CSV)

if not CFG.resume:
    now = datetime.now()
    EXP_NAME = now.strftime("%d-%m-%Y-%H-%M")
    print(f"Start experiment: {EXP_NAME}")

    writer = SummaryWriter(f"runs_v1/{EXP_NAME}")
else:
    # if resume should define from what to start
    PREV_NAME = CFG.prev_exp
    writer = SummaryWriter(f"runs_v1/{PREV_NAME}_{CFG.stage}")

LOSSES = {
    "cce": nn.CrossEntropyLoss(),
    "ls_soft_ce": LabelSmoothSoftmaxCEV1(),
}

# key - string, value - tuple(sceduler, if it epoch type)
epoch_type = True
SCHEDULERS = {
    "reduce_on_plateau": (ReduceLROnPlateau, epoch_type),
    "one_cycle": (OneCycleLR, not epoch_type),
    "cawr": (CosineAnnealingWarmRestarts, not epoch_type),
    "none": (None, None),
}


# Train Eval Loop
def train_eval_loop(
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    criterion,
    scheduler,
    sch_is_epoch_type,
    accum_step=CFG.accum_step,
    checkpoint=False,
    num_epochs=CFG.epoch,
    device=device,
    tb_tag="",
    model_name="debug",
):
    """Split it into the set of inner functions to siplify the loop itself"""
    # Inner Functions
    # write to TensorBoard helpers
    def weights_to_tb(step=0):
        conv1_weight = list(model.parameters())[0].data.to("cpu")
        img_grid = torchvision.utils.make_grid(conv1_weight.float(), normalize=True)
        writer.add_image(
            tag=f"Model conv1 weights {tb_tag}", img_tensor=img_grid, global_step=step
        )

    def input_img_to_tb(inputs, step):
        img = reverse_show_img(inputs[0], denorm=True)
        writer.add_image(
            tag=f"Input Image {tb_tag}",
            img_tensor=img,
            global_step=step,
            dataformats="HWC",
        )
        del img

    def preds_to_tb(outputs, inputs, labels, step):
        figure = plot_classes_preds(
            outputs.to("cpu"), inputs.to("cpu"), labels.to("cpu")
        )
        writer.add_figure(
            tag=f"Actuals vs Predictions {tb_tag}", figure=figure, global_step=step
        )

    def text_preds_to_tb(outputs, labels, step):
        preds_text = text_classes_preds(outputs.to("cpu"), labels.to("cpu"))
        writer.add_text(
            f"Actuals vs Predictions {tb_tag}", preds_text, global_step=step
        )

    def metrics_to_tb(mode, train_loss, train_score, val_loss, val_score, step):
        writer.add_text(
            f"On best {mode} save:",
            f"tr_loss: {train_loss:.4f}, tr_qwk: {train_score:.4f}, val_loss: {val_loss:.4f}, val_qwk: {val_score:.4f}",  # noqa
            global_step=step,
        )

    def conf_matrix_to_tb(val_epoch_labels, val_epoch_preds, step):
        writer.add_figure(
            tag=f"Confusion matrix {tb_tag}",
            figure=plot_confusion_matrix(
                val_epoch_labels, val_epoch_preds, normalize=True
            ),
            global_step=step,
        )

    # Train/Eval Loop
    # write first layer weights to TB @ init phase
    if not CFG.debug:
        weights_to_tb()

    # prepare model and optimizer
    model.to(device)
    if CFG.use_amp:  # automatic mixed precision
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    # define epochs numbers to look into input images and predictions,
    # no more than 10 times per full training
    vis_step = np.ceil(num_epochs / 10).astype(int)
    visual_epochs = list(range(0, num_epochs, vis_step))
    # metrics to wathch for model checkpointing
    best_qwk = -100 if not checkpoint else checkpoint["best_qwk"]
    best_val_loss = np.inf if not checkpoint else checkpoint["best_val_loss"]

    start_epoch = 0 if not checkpoint else checkpoint["epoch"] + 1
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("=" * 10)

        # Training Phase
        # Set training mode
        model.train()
        train_running_loss = 0.0
        train_epoch_preds, train_epoch_labels = [], []

        # We accumulate, zero at training epoch begins
        optimizer.zero_grad()

        # Iterate over train data.
        tk_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in tk_train:
            # Calculate global step for TensorBoard
            train_global_step = epoch * len(train_dataloader) + i

            inputs, labels = data
            # Visualize input before model at the middle of epoch:
            if epoch in visual_epochs and i == len(train_dataloader) // 2:
                input_img_to_tb(inputs, train_global_step)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if CFG.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # we accumulate gradients and make optimization step once per
            # # of accum_step
            if (i + 1) % accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            # loss is mean across batch, divide by number of steps in epoch
            # (so loss is normalized)
            train_running_loss += loss.item() / len(train_dataloader)
            # tensorboarding loss
            writer.add_scalar(
                tag=f"Training loss {tb_tag}",
                scalar_value=loss.item(),
                global_step=train_global_step,
            )

            # collect train preds and labels for QWK
            train_epoch_preds.append(outputs.data.to("cpu").numpy().argmax(1))
            train_epoch_labels.append(labels.to("cpu").numpy())
            # Add Batch Type Scheduler step here:
            if scheduler and not sch_is_epoch_type:
                scheduler.step()
        # Validation Phase
        # Set evaluation mode
        model.eval()
        val_running_loss = 0.0
        val_epoch_preds, val_epoch_labels = [], []
        # Iterate over val data
        tk_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for j, data in tk_val:
            # Calculate global step
            val_global_step = epoch * len(val_dataloader) + j
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() / len(val_dataloader)
            # tensorboarding loss
            writer.add_scalar(
                tag=f"Validation loss {tb_tag}",
                scalar_value=loss.item(),
                global_step=val_global_step,
            )

            # collect validation preds and labels for QWK
            val_epoch_preds.append(outputs.data.to("cpu").numpy().argmax(1))
            val_epoch_labels.append(labels.to("cpu").numpy())

            # visualise predictions for 0th validation batch
            if epoch in visual_epochs and j == 0:
                text_preds_to_tb(outputs, labels, val_global_step)

        # Epoch type Schedulers
        if scheduler and sch_is_epoch_type:
            scheduler.step(val_running_loss)
        # Write lr to TBD
        if CFG.finetune == "1stage":
            writer.add_scalar(
                tag=f"lr Interim {tb_tag}:",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=train_global_step,
            )
            writer.add_scalar(
                tag=f"lr Classifier {tb_tag}:",
                scalar_value=optimizer.param_groups[1]["lr"],
                global_step=train_global_step,
            )

        else:
            writer.add_scalar(
                tag=f"lr {tb_tag}:",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=train_global_step,
            )

        # "End of Epoch" Phase
        print(
            f"Training Loss: {train_running_loss:.4f}\tValidation Loss: {val_running_loss:.4f}"  # noqa
        )

        # Calculate epoch predictions distribution
        train_epoch_preds = np.concatenate(train_epoch_preds)
        train_epoch_labels = np.concatenate(train_epoch_labels)
        val_epoch_preds = np.concatenate(val_epoch_preds)
        val_epoch_labels = np.concatenate(val_epoch_labels)
        print(
            f"Counter train preds: {Counter(train_epoch_preds)}\tCounter val preds: {Counter(val_epoch_preds)}"  # noqa
        )
        # Calculate epoch QWK
        train_qwk = cohen_kappa_score(
            train_epoch_preds, train_epoch_labels, weights="quadratic"
        )
        val_qwk = cohen_kappa_score(
            val_epoch_preds, val_epoch_labels, weights="quadratic"
        )
        print(f"Epoch train QWK: {train_qwk:.3f}\tval QWK: {val_qwk:.3f}")
        writer.add_scalar(
            tag=f"Training QWK {tb_tag}", scalar_value=train_qwk, global_step=epoch
        )
        writer.add_scalar(
            tag=f"Validation QWK {tb_tag}", scalar_value=val_qwk, global_step=epoch
        )

        # On the best val loss do:
        if val_running_loss < best_val_loss:
            # update best and save model
            best_val_loss = val_running_loss
            best_qwk = val_qwk
            print(f"  Epoch {epoch} - Save Best Loss: {best_val_loss:.4f} Model")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_qwk": best_qwk,
                },
                f"{MODEL_PATH}/{model_name}_{epoch}_loss.pth",
            )
            # add losses as text to TB
            metrics_to_tb(
                "loss",
                train_running_loss,
                train_qwk,
                val_running_loss,
                val_qwk,
                val_global_step,
            )
            # add image of conv1 weights to TB
            if not CFG.debug:
                weights_to_tb(val_global_step)
            # add confusion matrix to TB
            conf_matrix_to_tb(val_epoch_labels, val_epoch_preds, val_global_step)

    # End of loop
    return model, best_val_loss, best_qwk


# Prepare CV - strategy, create cleaned in Suspisious Data Notebook
if CFG.debug:
    folds_fn = "folds_db.csv"
    try:
        folds = pd.read_csv(PANDA_PATH / folds_fn)
    except FileNotFoundError:
        folds = (
            TRAIN_DF.sample(n=100, random_state=CFG.seed).reset_index(drop=True).copy()
        )
else:
    folds_fn = "folds_cleaned.csv"
    try:
        folds = pd.read_csv(PANDA_PATH / folds_fn)
    except FileNotFoundError:
        folds = TRAIN_DF.copy()


if not (PANDA_PATH / folds_fn).exists():
    train_labels = folds[CFG.target_col].values
    kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_index, val_index) in enumerate(
        kf.split(folds.values, train_labels)
    ):
        folds.loc[val_index, "fold"] = int(fold)
    folds["fold"] = folds["fold"].astype(int)
    folds.to_csv(PANDA_PATH / folds_fn, index=None)
    folds.head()


# get folds (all experiments validated on fold 0)
train_df = folds[folds["fold"] != 0].copy()
val_df = folds[folds["fold"] == 0].copy()

# define datasets
if CFG.dataset == "lazy":
    train_ds = LazyTilesDataset(
        train_df, transform=get_transforms(data="train", aug=CFG.aug_type), debug=False
    )
    val_ds = TilesTrainDataset(
        val_df, is_train=False, transform=get_transforms(data="valid"), debug=False
    )  # same allways to compare with previous results
elif CFG.dataset == "tiles":
    train_ds = TilesTrainDataset(
        train_df,
        is_train=CFG.stoch_sample,
        transform=get_transforms(data="train", aug=CFG.aug_type),
        debug=False,
    )
    val_ds = TilesTrainDataset(
        val_df, is_train=False, transform=get_transforms(data="valid"), debug=False
    )
elif CFG.dataset == "patch":
    train_ds = PatchTrainDataset(
        train_df, transform=get_transforms(data="train", aug=CFG.aug_type), debug=False
    )
    val_ds = PatchTrainDataset(val_df, debug=False)
elif CFG.dataset == "hdf5":
    train_ds = H5PatchDataset(
        file_path=PANDA_PATH / "hdf5",
        fnames=[
            "patch256x16x1_fold_1.h5",
            "patch256x16x1_fold_2.h5",
            "patch256x16x1_fold_3.h5",
        ],
    )
    val_ds = H5PatchDataset(
        file_path=PANDA_PATH / "hdf5", fnames=["patch256x16x1_fold_0.h5"]
    )
else:
    print(f"No such dataset {CFG.dataset}")

# define a data loader
if CFG.dataset == "hdf5":
    # use specific sampler (so not to load hdf5 files to memory too frequently)
    sampler = SeqenceRandomSampler(len(train_ds), train_ds._common_len)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        sampler=sampler,
        num_workers=min(CFG.batch_size, 8),
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=min(CFG.batch_size, 8),
        pin_memory=True,
    )
else:
    train_dataloader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=min(CFG.batch_size, 8),
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=min(CFG.batch_size, 8),
        pin_memory=True,
    )
# Loss
criterion = LOSSES[CFG.loss]


# schedulers
def get_scheduler(
    optimizer, train_dataloader, schedule_type=CFG.schedule_type, resume=False
):
    assert schedule_type in SCHEDULERS, f"{schedule_type} not in SCHEDULERS"
    if schedule_type == "reduce_on_plateau":
        return (
            SCHEDULERS[schedule_type][0](
                optimizer,
                "min",
                factor=0.5,
                patience=CFG.rlopp if not resume else CFG.rlopp + 2,
                verbose=True,
            ),
            SCHEDULERS[schedule_type][1],
        )
    elif schedule_type == "one_cycle":
        return (
            SCHEDULERS[schedule_type][0](
                optimizer,
                max_lr=[CFG.lr, CFG.lr * 10] if CFG.finetune == "1stage" else CFG.lr,
                steps_per_epoch=len(train_dataloader),
                epochs=CFG.epoch,
                pct_start=0.05,
            ),
            SCHEDULERS[schedule_type][1],
        )
    elif schedule_type == "cawr":
        return (
            SCHEDULERS[schedule_type][0](
                optimizer, T_0=len(train_dataloader) * CFG.cawr_T, T_mult=CFG.cawr_Tmult
            ),
            SCHEDULERS[schedule_type][1],
        )
    else:
        return (SCHEDULERS[schedule_type][0], SCHEDULERS[schedule_type][1])


# Model init function
def init_model(train_df=train_df):
    model_ft = PatchModel(arch=CFG.arch)
    # initialize bias in the model
    cls_probas = (train_df[CFG.target_col].value_counts() / len(train_df)).values
    model_ft = init_last_layer_bias(model_ft, cls_probas)
    return model_ft


def init_optimizer(model_ft):
    if CFG.finetune == "1stage":
        freeze_botom(model_ft)
        interm_params = [
            p[1]
            for p in model_ft.named_parameters()
            if (not p[0].startswith("fc") and p[1].requires_grad)
        ]
        if CFG.optim == "adam":
            optimizer = torch.optim.Adam(
                [
                    {"params": interm_params, "lr": CFG.lr},
                    {"params": model_ft.fc.parameters(), "lr": CFG.lr * 10},
                ]
            )
        elif CFG.optim == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {"params": interm_params, "lr": CFG.lr},
                    {"params": model_ft.fc.parameters(), "lr": CFG.lr * 10},
                ],
                momentum=0.9,
                nesterov=True,
            )
    else:
        if CFG.optim == "adam":
            optimizer = torch.optim.Adam(
                model_ft.parameters(), lr=CFG.lr, amsgrad=False
            )
        elif CFG.optim == "sgd":
            optimizer = torch.optim.SGD(
                model_ft.parameters(), lr=CFG.lr, momentum=0.9, nesterov=True
            )
        elif CFG.optim == "radam":
            optimizer = RAdam(model_ft.parameters(), lr=CFG.lr)

    return optimizer


if CFG.debug:
    model_ft = init_model()
    optimizer = init_optimizer(model_ft)
    scheduler, sch_is_epoch_type = get_scheduler(optimizer, train_dataloader)

    model_ft, best_loss, best_qwk = train_eval_loop(
        train_dataloader,
        val_dataloader,
        model_ft,
        optimizer,
        criterion,
        scheduler,
        sch_is_epoch_type,
        num_epochs=3,
    )


if not CFG.debug and not CFG.resume:
    # Experiment
    writer.add_text("Experiment Description:", CFG.descript)
    model_ft = init_model()
    optimizer = init_optimizer(model_ft)
    scheduler, sch_is_epoch_type = get_scheduler(optimizer, train_dataloader)

    # Start Train/Eval Experiment
    model_ft, best_loss, best_qwk = train_eval_loop(
        train_dataloader,
        val_dataloader,
        model_ft,
        optimizer,
        criterion,
        scheduler,
        sch_is_epoch_type,
        model_name=EXP_NAME,
    )


if CFG.resume:
    # Resume Training
    checkpoint = torch.load(f"{MODEL_PATH}/{PREV_NAME}_{CFG.from_epoch}_loss.pth")

    model_ft = PatchModel(arch=CFG.arch)
    model_ft.load_state_dict(checkpoint["model_state_dict"])
    model_ft.to(device)
    optimizer = init_optimizer(model_ft)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # set smaller lr here
    for param_group in optimizer.param_groups:
        param_group["lr"] = CFG.lr

    scheduler, sch_is_epoch_type = get_scheduler(
        optimizer, train_dataloader, resume=True
    )

    # Start Train/Eval Experiment
    model_ft, best_loss, best_qwk = train_eval_loop(
        train_dataloader,
        val_dataloader,
        model_ft,
        optimizer,
        criterion,
        scheduler,
        sch_is_epoch_type,
        checkpoint=checkpoint,
        model_name=PREV_NAME,
    )

# After finish collect hyperparams used, best metrics and write to TensorBoard
hparam_dict = {
    key: val for key, val in CFG.__dict__.items() if not key.startswith("__")
}
metric_dict = {"hp/best_loss": best_loss, "hp/best_qwk": best_qwk}
writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

# Get the current git commit hash to add it in Tensorboard, to know exp code version
label = subprocess.check_output(["git", "describe", "--always"]).strip()
writer.add_text("Git commit hash:", label.decode())
