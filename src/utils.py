import torch
import torchvision
import cv2

import numpy as np

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
