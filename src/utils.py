import torch
import torchvision
import cv2

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score


MEAN = [0.485, 0.456, 0.406] * 4
STD = [0.229, 0.224, 0.225] * 4

# def denormalize(img_tensor):
#     img_tensor = img_tensor.clone()
#     for t, m, s in zip(img_tensor, MEAN, STD):
#         t.mul_(s).add_(m)
#     return img_tensor


def image_to_std_tensor(image, **params):
    image = torchvision.transforms.functional.to_tensor(image)
    image = torchvision.transforms.functional.normalize(image, MEAN, STD)
    return image




def f1_score_ravel(y_hat, y):
    return f1_score(np.ravel(y.cpu().double().numpy()), \
                    np.ravel(y_hat.cpu().softmax(1).argmax(1).double().numpy()))


def load_splits(folds_folder, val_folds=[0], train_folds=None):
    if isinstance(val_folds, int):
        val_folds = [val_folds,]

    folds = [int(fn.stem.split('_')[-1]) for fn in folds_folder.glob("fold_?.csv")]

    if train_folds is None:
        train_folds = [f for f in folds if f not in val_folds]

    if val_folds is None:
        train_folds = [f for f in folds if f not in train_folds]
    # print(val_folds, train_folds)
    val = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in val_folds])
    train = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in train_folds])
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, val


