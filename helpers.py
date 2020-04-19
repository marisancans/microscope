import torchvision
import torch
import numpy as np
from typing import List
import cv2


def pt_in_bb(bb, pt):
    x1, y1, x2, y2 = bb
    x, y = pt
    a = x1 < x < x2
    b = y1 < y < y2
    return a and b

def flatten_sequences(sequences_list):
    idxs = np.cumsum([len(x) for x in sequences_list])   
    catted = torch.cat(sequences_list, dim=0)
    return catted, idxs

def unflatten_sequences(catted, idxs):
    f_seq = []
    idxs = np.insert(idxs, 0, 0, axis=0)
    for i_from, i_to in zip(idxs, idxs[1:]):
        f_seq.append(catted[i_from:i_to])

    return f_seq