import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader

from models import TileSniperModel, DirectionModel, LSTMEncoder
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

tsm = TileSniperModel().to(args.device)
dm = DirectionModel(device=args.device).to(args.device)

loss_fn = torch.nn.MSELoss()
euclidean_loss = lambda a, b : torch.norm(a - b, 2)
optimizer = torch.optim.Adam(tsm.parameters(), 0.001)


@dataclass
class TileStack:
    tiles_imgs: np.array
    tiles_coords: tuple # coords relative to big image
    lines_coords: List # coords in this stack scope
    scores: torch.tensor = None
    coords: torch.tensor = None
    coords_scaled : torch.tensor = None


def stack_sniper(ts, imgs_t):
    scores, coords = tsm.forward(ts.tiles_imgs)
    ts.scores = scores
    ts.coords = coords

    # coords for tiles
    x1, y1, x2, y2 = ts.tiles_coords
    w = x2 - x1
    h = y2 - y1

    scaled = []
    for c_x, c_y in ts.coords:
        s_x = x1 + w * c_x
        s_y = y1 + w * c_y
        scaled.append(torch.stack([s_x, s_y]))
    
    ts.coords_scaled = torch.stack(scaled)

    return ts

def stack_iterator(n_portions, block_size, imgs_t, coords):
    for y in range(n_portions):
        for x in range(n_portions):
            y1 = y * block_size
            x1 = x * block_size
            y2 = (y + 1) * block_size
            x2 = (x + 1) * block_size
            tiles_imgs = imgs_t[:, :, y1:y2, x1:x2]

            bb = (x1, y1, x2, y2)
            lines_coords = []
            for stack_coords in coords:
                a = [c for c in stack_coords if pt_in_bb(bb, c)]
                lines_coords.append(a)
            
                if a:
                    a=0

            yield TileStack(tiles_imgs, (x1, y1, x2, y2), lines_coords)

def debug_stack_sniper(square, sniper_canvas, ts):
    score = float(ts.scores[0].cpu().detach())
    s = ((score + 1) / 2) 
    x1, y1, x2, y2 = ts.tiles_coords
    # print(s)
    
    square = cv2.rectangle(square, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (s), -1)

    op = 0.5
    overlay = cv2.addWeighted(sniper_canvas, op, square, 1-op, 0)

    if score > 0:
        xc, yc = ts.coords_scaled[0]
        sniper_canvas = cv2.circle(sniper_canvas, (int(xc), int(yc)), 10, (1), 2)

    cv2.namedWindow("stack_sniper", cv2.WINDOW_NORMAL)
    cv2.imshow("stack_sniper", overlay)
    cv2.waitKey(1)

def snake_walker(ts, coords_idxs, debug_canvas, imgs_t):    
    snake_canvas = debug_canvas.copy()

    picked_imgs_t = [imgs_t[i] for i in coords_idxs]
    x = [int(ts.coords_scaled[i][0]) for i in coords_idxs]
    y = [int(ts.coords_scaled[i][1]) for i in coords_idxs]
    x = np.array(x)
    y = np.array(y)
    
    memories = [[] for i in coords_idxs]
    all_angles = []
    all_x = []
    all_y = []
    
    half_patch = patch_size // 2
    clamp = lambda a : np.clip(a, half_patch, img_size - half_patch)

    for n in range(10):
        x = clamp(x)
        y = clamp(y)
        
        all_x.append(x)
        all_y.append(y)

        # pick patches
        for ith_x, ith_y, img_t, mem, ci in zip(x, y, picked_imgs_t, memories, coords_idxs):
            x1 = ith_x - half_patch
            y1 = ith_y - half_patch

            x2 = x1 + patch_size
            y2 = y1 + patch_size

            patch = img_t[:, y1:y2, x1:x2]
            mem.append(patch)

            if ci == 0:
                snake_canvas = cv2.rectangle(snake_canvas, (x1, y1), (x2, y2), (1), 2)
                cv2.namedWindow("snake_walker", cv2.WINDOW_NORMAL)
                cv2.imshow("snake_walker", snake_canvas)
                cv2.waitKey(1)


    
        # take only last n patches
        lengths = [np.clip(args.lstm_memory, 0, len(m)) for m in memories]
        sequenced_patches = [torch.stack(m[-l:]) for m, l in zip(memories, lengths)]

        flattened_patches, flat_ids = flatten_sequences(sequenced_patches) 

        scores, angles = dm.forward(flattened_patches, flat_ids)
        
        # remove last dim
        scores = scores.squeeze(-1)
        angles = angles.squeeze(-1)
        
        angles_np = angles.cpu().detach().numpy()

        # scale angle to radians
        angles_np = angles_np * np.pi * 2
        all_angles.append(angles_np)

        x = x + (patch_magnitude * np.cos(angles_np)).astype(int)
        y = y + (patch_magnitude * np.sin(angles_np)).astype(int)

    return memories, all_angles, all_x, all_y







while True:
    for imgs_t, coords in dataloader:
        imgs_t = imgs_t.to(args.device)
        # debug
        img_t = imgs_t[0].cpu()
        debug_canvas = img_t.squeeze().numpy()
        square = np.zeros_like(debug_canvas)
        sniper_canvas = debug_canvas.copy()

        for ts in stack_iterator(n_portions, block_size, imgs_t, coords):
            ts = stack_sniper(ts, imgs_t)

            # debug
            if args.debug:
                debug_stack_sniper(square, sniper_canvas, ts)

            # loss for tile sniper
            scorings_truth = [torch.tensor(1) if lc else torch.tensor(-1) for lc in ts.lines_coords]
            scorings_truth = torch.stack(scorings_truth).unsqueeze(-1).float()
            scorings_truth = scorings_truth.to(args.device)

            # loss for snake walker
            coords_preds  = []
            coords_truths = []
            coords_idxs   = []

            for i, (lc, cs) in enumerate(zip(ts.lines_coords, ts.coords_scaled)):
                if not lc:
                    continue
                coords_truths.append(lc[len(lc) // 2])
                coords_preds.append(cs)
                coords_idxs.append(i)

            direction_loss = torch.tensor(0)
            # if coords_idxs:
            #     memories, angles_np, all_x, all_y = snake_walker(ts, coords_idxs, debug_canvas, imgs_t)

            #     lcs = [ts.lines_coords[i] for i in coords_idxs]
            #     c_scaled = [ts.coords_scaled.detach().cpu().numpy() for i in coords_idxs]

            #     euclids = []
            #     for line, s in zip(lcs, c_scaled):
            #         e = [np.linalg.norm(pt - s) for pt in line]
            #         e = sorted(e)
            #         euclids.append(e)
                
            #     lengths = [np.clip(n_pts_direction_vector, 0, len(e)) for e in euclids]
            #     p = [e[-l:] for e, l in zip(euclids, lengths)]

            #     a=0


            coords_loss = torch.tensor(0)
            if coords_preds and coords_truths:
                coords_truths = torch.tensor(coords_truths).to(args.device).float()
                coords_preds = torch.stack(coords_preds)

                coords_loss = euclidean_loss(coords_truths, coords_preds)

                 
            
            loss = loss_fn(ts.scores, scorings_truth) + coords_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)












