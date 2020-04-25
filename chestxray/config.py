import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH"))

# those precomputed in C1M1 notebook
CHESTRAIN_MEAN = 122.40529296875
CHESTRAIN_STD = 70.33326676182992
