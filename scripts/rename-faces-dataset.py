import sys,os
sys.path.append(os.getcwd())

from os import rename
from pathlib import Path
from fire import Fire
from tqdm.cli import tqdm

def main(folder):
    folder = Path(folder)
    img_filenames = folder.glob("**/*.png")
    for img_fn in tqdm(img_filenames, desc="rename files"):
        rename(
            str(img_fn),
            str(img_fn).replace("=", "-"),
        )

if __name__ == "__main__":
    Fire(main)

