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
    img_height = 1024
    img_width = 1024
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 16
    tile_sz = 256
    batch_size = 4
    accum_step = 16  # effective batch size will be batch_size * accum_step
    dataset = "hdf5"  # "patch", "tiles", "lazy", "hdf5"
    aug_type = "light"  # "light" or "heavy"
    # model
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    schedule_type = "none"  # "one_cycle", "reduce_on_plateau" or "cawr"
    cawr_T = 1
    cawr_Tmult = 2
    # loss
    loss = "cce"  # "cce" or "ls_soft_ce"
    # optim
    optim = "adam"  # "adam", "sgd" or "radam"
    lr = 0.001 if optim == "sgd" else 1e-4
    # schedule
    rlopp = 3  # learnig rate on plateu scheduler patience
    # training
    epoch = 70
    n_fold = 4
    use_amp = True
