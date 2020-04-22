import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader

from models import UNet
from datasets import SynthDataset, collate_fn

from helpers import pt_in_bb, flatten_sequences

from collections import deque
import argparse

def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--debug', type=arg_to_bool, default=False)
parser.add_argument('--lstm_memory', type=int, default=4)
args = parser.parse_args()


# params
img_size = 500
n_portions = 10
block_size = img_size // n_portions # 50

patch_magnitude = 25.0
patch_size = 50

n_pts_direction_vector = 3

dataset = SynthDataset(batch_size=args.batch_size, img_size=img_size)
dataloader = DataLoader(dataset=dataset, num_workers=0, collate_fn=collate_fn)

unet = UNet().to(args.device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), 0.001)


def stack_iterator(n_portions, block_size, imgs_t):
    for y in range(n_portions):
        for x in range(n_portions):
            y1 = y * block_size
            x1 = x * block_size
            y2 = (y + 1) * block_size
            x2 = (x + 1) * block_size
            tile_imgs = imgs_t[:, :, y1:y2, x1:x2]

            yield tile_imgs


while True:
    for imgs_t, layers in dataloader:
        imgs_t = imgs_t.to(args.device)
        # debug
        img_t = imgs_t[0].cpu()
        debug_canvas = img_t.squeeze().numpy()
        square = np.zeros_like(debug_canvas)
        sniper_canvas = debug_canvas.copy()

        for ts in stack_iterator(n_portions, block_size, imgs_t):
            pass



                 
            
            # loss = loss_fn(ts.scores, scorings_truth)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # print(loss)












