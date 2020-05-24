import multiprocessing
from pathlib import Path

import numpy as np
import skimage.io
import tqdm
from PIL import Image

from chestxray.config import PANDA_IMGS


def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    (ys,) = (image.min((1, 2)) < value).nonzero()
    (xs,) = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def to_jpeg(path: Path):
    image = skimage.io.MultiImage(str(path))
    image_to_jpeg(path, "_1", image[1])


def image_to_jpeg(path: Path, suffix: str, image: np.ndarray):
    jpeg_path = path.parent / f"{path.stem}{suffix}.jpeg"
    image = crop_white(image)
    image = Image.fromarray(image)
    image.save(jpeg_path, quality=90)


def main():
    paths = list(PANDA_IMGS.glob("*.tiff"))[:500]
    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(pool.imap(to_jpeg, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    main()
