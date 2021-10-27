import torch
import torchvision
import cv2

import numpy as np
import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet

class VideoFramesJoint(torch.nn.Module):
    def __init__(self, model_name="efficientnet-b0", in_channels=12, num_classes=2):
        super().__init__()

        self.encoder = EfficientNet.from_pretrained(model_name, in_channels=3)
        layers_to_remove = ["_fc", "_swish"]
        for l in layers_to_remove:
            setattr(self.encoder, l, torch.nn.Identity())

        self.in_images = in_channels // 3
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.in_images * 1280, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        encoded = []
        for i in range(self.in_images):
            img = x[:, i*3:i*3+3, :, :]
            encoded.append(self.encoder(img))

        features = torch.cat(encoded, 1)
        out = self.head(features)
        return out


class ImageStackClassificationModel(pl.LightningModule):
    # TODO why models_kwargs (not just model_kwargs)
    def __init__(self,
                 model_class,
                 models_kwargs,
                 optimizer_class,
                 optimizer_kwargs,
                 scheduler_class,
                 scheduler_kwargs,
                 criterion,
                 monitor,
                 metrics,):

        super().__init__()
        self.save_hyperparameters()
        pl.seed_everything(42)

        self.model = model_class(**models_kwargs)
        self.optimizer = optimizer_class(self.model.parameters(),
                                         **optimizer_kwargs)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

        self.criterion = criterion
        self.monitor = monitor
        self.metrics = metrics


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        outs = self(imgs)
        loss = self.criterion(outs, targets)
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        outs = self(imgs)
        loss = self.criterion(outs, targets)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 logger=True)

        scores = {}
        for metric_name, metric in  self.metrics.items():
            scores[f"val_{metric_name}"] = metric(outs, targets)

        self.log_dict(scores,
                      prog_bar=False,
                      on_step=False,
                      on_epoch=True,
                      logger=True)
        return loss

    def configure_optimizers(self):
        scheduler_info = {
            "scheduler": self.scheduler,
            "monitor": self.monitor,
            'interval': 'epoch',
        }
        return [self.optimizer], [scheduler_info]

