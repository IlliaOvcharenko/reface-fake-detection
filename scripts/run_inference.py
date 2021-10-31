import sys,os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
import cv2

import numpy as np
import pandas as pd
import albumentations as A
import pytorch_toolbelt.losses as L
import pytorch_lightning as pl

from fire import Fire
from pathlib import Path
from tqdm.cli import tqdm
from efficientnet_pytorch import EfficientNet

from src.data import (ImageStackDataset,
                      ImageDataModule)
from src.model import ImageStackClassificationModel
from src.utils import (MEAN, STD,
                       image_to_std_tensor,
                       f1_score_ravel)


def predict_with_dataloader(expr, dataloader):
    expr.eval();
    result_df = pd.DataFrame()

    with torch.no_grad():
        for names, imgs in tqdm(dataloader):
            outs = expr(imgs)
            probs = outs.detach().softmax(1)
            outs = probs.argmax(1).numpy()
            for fn, out in zip(names, outs):
                result_df = result_df.append({
                    "filename": fn,
                    "label": out,
                }, ignore_index=True)

    return result_df


def main(
    model_ckpt_fn,
    save_preds_fn,
):
    expr = ImageStackClassificationModel.load_from_checkpoint(
        model_ckpt_fn,
    )

    data_folder = Path("data")
    test_folder = data_folder / "frames" / "test"
    test_df = pd.DataFrame({"filename": [f"{fn.stem}.mp4" for fn in  test_folder.glob("*")]})

    test_dataset = ImageStackDataset(
        test_df, test_folder,
        "test",
        A.Compose([A.Resize(256, 256), image_to_std_tensor]),
        resize=None,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=6
    )

    test_preds_df = predict_with_dataloader(expr, test_dataloader)
    test_preds_df["label"] = test_preds_df["label"].astype(int)
    test_preds_df.to_csv(save_preds_fn, index=False)



if __name__ == "__main__":
    Fire(main)

