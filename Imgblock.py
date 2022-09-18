import numpy as np
import random
import torch

def TensorBlock(x, M):
    size = x.shape[-1]
    step = int(size / M)
    result = torch.zeros(x.shape)
    orderlist = [i for i in range(M * M)]
    random.shuffle(orderlist)

    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch = x[:, :, int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        order = orderlist[i]
        A, B = (order // M, order % M)
        a, b = (A * step, B * step)
        result[:, :, int(a) : int(a) + step, int(b) : int(b) + step] = patch
    return result


def BlockSegment(x, M):
    """
    M must be a factor of size of x
    """
    size = x.shape[0]
    step = int(size / M)
    result = np.zeros(shape = x.shape)
    orderlist = [i for i in range(M * M)]
    random.shuffle(orderlist)
    print(orderlist)
    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch = x[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        order = orderlist[i]
        A, B = (order // M, order % M)
        a, b = (A * step, B * step)
        result[int(a) : int(a) + step, int(b) : int(b) + step] = patch
    return result


def RGBSegment(x, M):
    size = x.shape[0]
    step = int(size / M)
    result = np.zeros(shape = x.shape)
    orderlist = [i for i in range(M * M)]
    random.shuffle(orderlist)
    #print(orderlist)
    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch = x[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step, 0:]
        order = orderlist[i]
        A, B = (order // M, order % M)
        a, b = (A * step, B * step)
        result[int(a) : int(a) + step, int(b) : int(b) + step, 0:] = patch
    return result


