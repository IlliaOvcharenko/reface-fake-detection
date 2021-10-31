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
                       f1_score_ravel)


def test_dataset():
    df = pd.read_csv("data/train.csv")
    ds = ImageStackDataset(df, Path("data/frames/train"), "train",
                           A.Compose([A.Resize(128, 128),]))
    print(ds[0])


def test_model():
    model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=12, num_classes=2)
    print(model)


def init_training():
    data_folder = Path("data")

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        image_to_std_tensor,
    ])

    val_test_transform = A.Compose([
        A.Resize(256, 256),
        image_to_std_tensor,
    ])


    dm = ImageDataModule(
        pd.read_csv(data_folder / "splits" / "train.csv"),
        data_folder / "frames" / "train",
        train_transform,
        64,
        {},

        pd.read_csv(data_folder / "splits" / "val.csv"),
        data_folder / "frames" / "train",
        val_test_transform,
        128,
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
        criterion=torch.nn.CrossEntropyLoss(),
        # criterion=L.FocalLoss(),
        metrics={
            "f1": f1_score_ravel,
        }
    )


    models_folder = Path("models")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=models_folder,
        filename="{epoch}-{step}-{val_f1:.4f}",
        mode="max",
        monitor="val_f1",
        save_top_k=5
    )

    logs_folder = Path("logs")
    tb_logger = pl.loggers.TensorBoardLogger(logs_folder)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        gpus=1,
        max_epochs=300,
        deterministic=True,

        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        # val_check_interval=0.5,
    )

    trainer.fit(expr, dm)


def main():
    # test_dataset()
    # test_model()

    init_training()


if __name__ == "__main__":
    Fire(main)

