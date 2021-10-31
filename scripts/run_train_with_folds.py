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
from efficientnet_pytorch import EfficientNet

from src.data import (ImageStackDataset,
                      ImageDataModule)
from src.model import (VideoFramesJoint,
                       ImageStackClassificationModel)
from src.utils import (MEAN, STD,
                       image_to_std_tensor,
                       f1_score_ravel,
                       load_splits)


def init_training(train_df, val_df, model_name):

    train_transform = A.Compose([
        # A.ShiftScaleRotate(shift_limit=0.1,
        #                    scale_limit=0.05,
        #                    rotate_limit=5,
        #                    p=0.3,
        #                    border_mode=1),
        A.Resize(256, 256),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        image_to_std_tensor,
    ])

    val_test_transform = A.Compose([
        A.Resize(256, 256),
        image_to_std_tensor,
    ])


    data_folder = Path("data")
    dm = ImageDataModule(
        train_df,
        data_folder / "frames" / "train",
        train_transform,
        64,
        # {"resize": (256, 256)},
        {},

        val_df,
        data_folder / "frames" / "train",
        val_test_transform,
        128,
        # {"resize": (256, 256)},
        {},
    )

    expr = ImageStackClassificationModel(
        EfficientNet.from_pretrained,
        {"model_name": "efficientnet-b0", "in_channels": 12, "num_classes": 2},

        torch.optim.Adam,
        {"lr": 0.01},

        torch.optim.lr_scheduler.ReduceLROnPlateau,
        {"patience": 10, "mode": "max", "factor": 0.6},


        monitor="val_f1",

        # criterion=L.FocalLoss(),
        criterion=torch.nn.CrossEntropyLoss(),
        metrics={
            "f1": f1_score_ravel,
        }
    )


    model_folder = Path("models") / model_name
    model_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_folder,
        filename="{epoch}-{step}-{val_f1:.4f}",
        mode="max",
        monitor="val_f1",
        save_top_k=5
    )

    logs_folder = Path("logs")
    tb_logger = pl.loggers.TensorBoardLogger(logs_folder, name=model_name)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        gpus=1,
        max_epochs=50,
        deterministic=True,

        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        # val_check_interval=1,
    )

    trainer.fit(expr, dm)


def init_training_with_folds(
        folds_folder
):
    folds_folder = Path(folds_folder)
    n_folds = len(list(folds_folder.glob("*.csv")))
    print(f"Folds folder: {folds_folder}, num folds: {n_folds}")

    for fold_idx in range(n_folds):
        print(f"Train fold: {fold_idx}")
        tr_df, val_df = load_splits(folds_folder, val_folds=fold_idx)
        model_name = f"fold-{fold_idx}"
        init_training(tr_df, val_df, model_name)


if __name__ == "__main__":
    Fire(init_training_with_folds)

