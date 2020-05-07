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


# PANDA competition config
class CFG:
    debug = False
    use_amp = True
    img_height = 512
    img_width = 512
    lr = 1e-4
    batch_size = 16
    epoch = 30
    seed = 1982
    target_size = 6
    img_id_col = "image_id"
    target_col = "isup_grade"
    n_fold = 4
    tiff_layer = -1
