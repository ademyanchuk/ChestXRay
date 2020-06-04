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
    debug = True
    seed = 1982
    # data
    img_height = 256
    img_width = 256
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 32
    tile_sz = 256
    batch_size = 2
    accum_step = 2  # effective batch size will be batch_size * accum_step
    dataset = "patch"  # "patch", "tiles", "lazy", "hdf5"
    aug_type = "heavy"  # "light" or "heavy"
    # model
    arch = "resnet50"  # "resnet34", "resnet50", "bitM"
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    # loss
    loss = "cce"  # "cce" or "ls_soft_ce"
    # optim
    optim = "radam"  # "adam", "sgd" or "radam"
    lr = 0.001 if optim == "sgd" else 1e-4
    # schedule
    schedule_type = "none"  # "one_cycle", "reduce_on_plateau" or "cawr"
    cawr_T = 1
    cawr_Tmult = 2
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    resume = False
    prev_exp = None
    from_epoch = None
    stage = None
    epoch = 35
    n_fold = 4
    use_amp = True
    # Experiment
    descript = "Debug"
