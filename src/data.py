import torch
import torchvision
import cv2

import numpy as np
import pandas as pd
import albumentations as A
import pytorch_lightning as pl

from pathlib import Path
from efficientnet_pytorch import EfficientNet


class ImageStackDataset(torch.utils.data.Dataset):
    def __init__(self, df, folder, mode, transform=None, resize=None):
        self.df = df
        self.folder = folder
        self.mode = mode
        self.transform = transform
        self.resize = resize


    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        video_folder  = self.folder / item["filename"].replace(".mp4", "")
        # print(video_folder, video_folder.exists(), len(list(video_folder.glob("*"))))
        # print(video_folder)
        imgs = [cv2.imread(str(img_fn)) for img_fn in video_folder.glob("*.png")]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        if self.resize is not None:
            imgs = [cv2.resize(img, self.resize) for img in imgs]

        if len(imgs) == 0:
            imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(4)]
        if len(imgs) < 4:
            imgs = imgs * 4
        # TODO fix frame cropping and get rid of this hack
        if len(imgs) > 4:
            imgs = imgs[:4]

        imgs = np.concatenate(imgs, -1)


        if self.transform is not None:
            transformed = self.transform(image=imgs)
            imgs = transformed

        if self.mode in ["test"]:
            return item["filename"], imgs

        elif self.mode in ["train", "val"]:
            target = item["label"]
            target = torch.tensor(target).long()
            return imgs, target

    def __len__(self):
        return len(self.df)


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df, train_folder, train_transform, train_batch_size,
        train_ds_params,
        val_df, val_folder, val_transform, val_batch_size,
        val_ds_params,
        num_workers=6,
    ):
        super().__init__()
        self.train_df = train_df
        self.train_folder = train_folder
        self.train_transform = train_transform
        self.train_batch_size = train_batch_size
        self.train_ds_params = train_ds_params

        self.val_df = val_df
        self.val_folder = val_folder
        self.val_transform = val_transform
        self.val_batch_size = val_batch_size
        self.val_ds_params = val_ds_params

        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = ImageStackDataset(
            self.train_df,
            self.train_folder,
            "train",
            self.train_transform,
            **self.train_ds_params,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = ImageStackDataset(
            self.val_df,
            self.val_folder,
            "val",
            self.val_transform,
            **self.val_ds_params,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return dataloader

