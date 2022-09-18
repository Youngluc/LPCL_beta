import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision import transforms
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
import random
import Imgblock
import numpy as np
from datetime import datetime
import sklearn.metrics

def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def euclidean_distances(x, y, squared=True):
    """Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.expand_dims(np.einsum('ij,ij->i', y, y), axis=0)
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


import cv2
import saliency
import MutuInfo

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def new_Sortbysocre(x, M):
    temp = x.cpu()
    resized = tensor_to_np(temp)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sal = saliency.LC(gray)
    ha = []
    step = int(224 / M)
    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch_gray = sal[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        patch_rgb = resized[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step, 0:]
        patch = gray[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        #ha.append((MutuInfo.Entropy(patch), np.sum(patch_gray), patch, i))
        ha.append((np.mean(MutuInfo.RGBEntropy(patch_rgb)), np.sum(patch_gray), patch_rgb, i))
        #ha.append((MutuInfo.Entropy(patch_gray), patch1, i))
    sorted_res = sorted(ha, key = lambda x : x[0], reverse = True)[0: (M**2) // 2]
    sorted_res.sort(key = lambda x : x[1], reverse = True)
    return np.array(sorted_res[0 : len(sorted_res) // 2], dtype = object)



if __name__ == '__main__':
    x1 = np.random.rand(10, 32*32)
    x2 = np.random.rand(5, 32*32)
    res = euclidean_distances(x1, x2)
    print(res.argmax(axis = 1))
    

    