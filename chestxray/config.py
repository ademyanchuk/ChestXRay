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
    img_height = 224
    img_width = 224
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 36
    tile_sz = 224
    batch_size = 8
    accum_step = 1  # effective batch size will be batch_size * accum_step
    dataset = "patch"  # "patch", "tiles", "lazy", "hdf5"
    aux_tile = False  # for Tiles Dataset
    aux_tile_sz = 0  # squares produced from both tile sizes need to be same size
    aux_tile_num = 0  # see above
    aug_type = "heavy"  # "light" or "heavy"
    # model
    att = True  # use attention for MIL-pooling, only for patch
    arch = "resnet34"  # "resnet34", "resnet50", "bitM", "efnet"
    enet_bone = "efficientnet-b0"
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    pre_init_fc_bias = False
    # loss
    ohem = True  # will work with ohem and bce
    loss = "bce"  # "cce" or "ls_soft_ce", "ohem", "bce"
    # optim
    optim = "radam"  # "adam", "adamw" "sgd" or "radam"
    lr = 1e-3 if optim == "sgd" else 3e-4
    wd = 1e-2
    # schedule
    schedule_type = "cawr"  # "one_cycle", "reduce_on_plateau" or "cawr"
    oc_final_div_factor = 1e1
    cawr_T_0 = 10  # epochs untill first restart
    cawr_T_mult = 2  # multiply next restarts
    cawr_T_up = 3  # warmup epochs
    cawr_gamma = 0.8
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    resume = False
    prev_exp = "None"
    from_epoch = 0
    stage = 0
    epoch = 70
    n_fold = 4
    use_amp = True
    # Experiment
    descript = "bce-ohem + rn34 + 224x36 patch-att + CAWUR"
