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
    tile_sz = 128
    batch_size = 6
    dataset = "patch"  # "patch", "tiles" or "lazy"
    aug_type = "light"  # "light" or "heavy"
    # model
    finetune = False  # or "1stage"
    model_cls = "one_layer"  # "one_layer" or "deep"
    schedule_type = "reduce_on_plateau"  # "one_cycle", "reduce_on_plateau" or "cawr"
    cawr_T = 1
    cawr_Tmult = 2
    # loss
    loss = "ls_soft_ce"
    # optim
    optim = "radam"  # "adam", "sgd" or "radam"
    lr = 0.1 * batch_size / 512 if optim == "sgd" else 3e-4
    # schedule
    rlopp = 1  # learnig rate on plateu scheduler patience
    # training
    epoch = 70
    n_fold = 4
    use_amp = True
