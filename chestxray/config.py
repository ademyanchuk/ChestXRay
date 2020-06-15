import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH"))

PROJ_PATH = DATA_PATH.parent

MODEL_PATH = PROJ_PATH / "models"

PANDA_PATH = DATA_PATH / "Panda"
PANDA_IMGS = PANDA_PATH / "train_images"
PANDA_MASKS = PANDA_PATH / "train_label_masks"
TRAIN_CSV = PANDA_PATH / "train.csv"

TILES_IMGS = PANDA_PATH / "tiles36x128x1"  # "tiles144x64x1" or "tiles36x128x1"


# PANDA competition config
class CFG:
    # overall
    debug = False
    seed = 1982
    # data
    img_height = 256
    img_width = 256
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 36
    tile_sz = 256
    batch_size = 4
    accum_step = 1  # effective batch size will be batch_size * accum_step
    dataset = "patch"  # "patch", "tiles", "lazy", "hdf5"
    aug_type = "light"  # "light" or "heavy"
    # model
    arch = "resnet34"  # "resnet34", "resnet50", "bitM", "efnet"
    enet_bone = "efficientnet-b0"
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    pre_init_fc_bias = False
    # loss
    loss = "huber"  # "cce" or "ls_soft_ce", "ohem", "bce", "huber"
    # optim
    optim = "radam"  # "adam", "sgd" or "radam"
    lr = 1e-3 if optim == "sgd" else 3e-4
    # schedule
    schedule_type = "one_cycle"  # "one_cycle", "reduce_on_plateau" or "cawr"
    cawr_T = 1
    cawr_Tmult = 2
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    resume = False
    prev_exp = "None"
    from_epoch = 0
    stage = 0
    epoch = 33
    n_fold = 4
    use_amp = True
    # Experiment
    descript = "huber + rn34 + one cycle + 256x36"
