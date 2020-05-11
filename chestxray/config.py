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
    debug = False
    use_amp = True
    img_height = 768
    img_width = 768
    lr = 1e-4 * 0.5
    batch_size = 8
    epoch = 30
    seed = 1982
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    n_fold = 4
    tiff_layer = 1
    loss = "ls_soft_ce"
    stoch_sample = True
