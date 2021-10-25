import cv2

from fire import Fire
from pathlib import Path


def main(folder):
    folder = Path(folder)
    for img_fn in folder.glob("**/*.png"):
        img = cv2.imread(str(img_fn))
        print(img.shape)


if __name__ == "__main__":
    Fire(main)
