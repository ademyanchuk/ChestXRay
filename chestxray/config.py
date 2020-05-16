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

TILES_IMGS = PANDA_PATH / "tiles144x64x1"


# PANDA competition config
class CFG:
    # overall
    debug = False
    seed = 1982
    # data
    img_height = 768
    img_width = 768
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    tiff_layer = 1
    stoch_sample = True
    num_tiles = 144
    tile_sz = 64
    batch_size = 8
    use_lazy = True
    aug_type = "light"
    # model
    finetune = False  # or "1stage"
    model_cls = "deep"  # "one_layer" or "deep"
    schedule_type = "cawr"  # "one_cycle", "reduce_on_plateau" or "cawr"
    cawr_T = 1
    cawr_Tmult = 2
    # loss
    loss = "ls_soft_ce"
    # optim
    optim = "sgd"  # "adam" or "sgd"
    lr = 0.1 * batch_size / 512 if optim == "sgd" else 5e-5
    # schedule
    rlopp = 3  # learnig rate on plateu scheduler patience
    # training
    epoch = 50
    n_fold = 4
    use_amp = True
