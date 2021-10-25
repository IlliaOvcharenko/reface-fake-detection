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
    def __init__(self, df, folder, mode, transform=None):
        self.df = df
        self.folder = folder
        self.mode = mode
        self.transform = transform


    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        video_folder  = self.folder / item["filename"].replace(".mp4", "")
        # print(video_folder)
        imgs = [cv2.imread(str(img_fn)) for img_fn in video_folder.glob("*.png")]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        # TODO fix frame cropping and get rid of this hack
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
        val_df, val_folder, val_transform, val_batch_size,
        num_workers=6,
    ):
        super().__init__()
        self.train_df = train_df
        self.train_folder = train_folder
        self.train_transform = train_transform
        self.train_batch_size = train_batch_size

        self.val_df = val_df
        self.val_folder = val_folder
        self.val_transform = val_transform
        self.val_batch_size = val_batch_size

        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = ImageStackDataset(self.train_df, self.train_folder, "train", self.train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = ImageStackDataset(self.val_df, self.val_folder, "val", self.val_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return dataloader

