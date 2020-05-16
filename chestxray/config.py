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

TILES_IMGS = PANDA_PATH / "tiles32x128x1"


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
    num_tiles = 36
    tile_sz = 128
    batch_size = 16
    use_lazy = True
    aug_type = "light"
    # model
    finetune = "1stage"
    model_cls = "deep"
    schedule_type = "one_cycle"
    # loss
    loss = "ls_soft_ce"
    # optim
    lr = 5e-5
    # schedule
    rlopp = 3  # learnig rate on plateu scheduler patience
    # training
    epoch = 70
    n_fold = 4
    use_amp = True
