import cv2
import numpy as np

# from unet import UNet

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List
from line_gen import SquiggleGen
from collections import OrderedDict
import torch.nn.functional as F
from helpers import pt_in_bb

import argparse

def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()

@dataclass
class SynthDataset(Dataset):
    batch_size: int = 64
    img_size: int = 1000
    start_pos: Tuple[int, int] = None

    def __post_init__(self):
        if not self.start_pos:
            self.start_pos = (self.img_size // 2, self.img_size // 2)
        self.sg = SquiggleGen(size = self.img_size, debug=False)

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        imgs, coords = self.sg.gen_imgs(self.batch_size, self.start_pos)
        # offset

        imgs_t = torch.tensor(imgs).float()
        imgs_t = imgs_t.unsqueeze(1) # Add channel dimennsion
        return imgs_t, coords

def collate_fn(sub_batches):
    b_imgs = []
    b_coords = []
    
    for imgs_t, coords in sub_batches:
        b_imgs.append(imgs_t)
        b_coords.extend(coords) 
    
    return torch.cat(b_imgs), np.array(b_coords)



def encoder_block(in_channels, features, name):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class TileSniperModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(TileSniperModel, self).__init__()

        # encoder
        self.encoder1 = encoder_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = encoder_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_score = nn.Linear(in_features=features * 2, out_features=1)
        self.fc_coords = nn.Linear(in_features=features * 2, out_features=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = F.relu(enc1)

        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = F.relu(enc2)

        avg_pooled = self.avg_pool(enc2)
        avg_pooled = avg_pooled.squeeze(-1).squeeze(-1)

        score = self.fc_score(avg_pooled)
        coords = self.fc_coords(avg_pooled)

        score = F.tanh(score)
        coords = F.sigmoid(coords)

        return score, coords


class DirectionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(DirectionModel, self).__init__()

        # encoder
        self.encoder1 = encoder_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc1 = nn.BatchNorm2d(features)
        self.encoder2 = encoder_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc2 = nn.BatchNorm2d(features * 2)
        self.encoder3 = encoder_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc3 = nn.BatchNorm2d(features * 4)
        self.encoder4 = encoder_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc4 = nn.BatchNorm2d(features * 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc_score = nn.Linear(in_features=features * 8, out_features=1)
        self.fc_angle = nn.Linear(in_features=features * 8, out_features=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = self.bn_enc1(enc1)
        enc1 = F.relu(enc1)

        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.bn_enc2(enc2)
        enc2 = F.relu(enc2)

        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3)
        enc3 = self.bn_enc3(enc3)
        enc3 = F.relu(enc3)

        enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc4)
        enc4 = self.bn_enc4(enc4)
        enc4 = F.relu(enc4)

        avg_pooled = self.avg_pool(enc4)
        avg_pooled = avg_pooled.squeeze(-1).squeeze(-1)

        score = self.fc_score(avg_pooled)
        angle = self.fc_angle(avg_pooled)

        score = F.tanh(score)
        angle = F.sigmoid(angle)

        return score, angle



# params
img_size = 500
n_portions = 10
block_size = img_size // n_portions # 50

patch_magnitude = 10
ptch_size = 50

dataset = SynthDataset(batch_size=args.batch_size, img_size=img_size)
dataloader = DataLoader(dataset=dataset, num_workers=0, collate_fn=collate_fn)


tsm = TileSniperModel().to(args.device)
dm = DirectionModel()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(tsm.parameters(), 0.001)

# # cv2.imshow('img', img)
# # cv2.waitKey(1)


# # img = np.zeros((400, 300), dtype=np.uint8)
# orignal = cv2.imread("pic.png")
# orignal = cv2.cvtColor(orignal, cv2.COLOR_BGR2GRAY)

@dataclass
class TileStack:
    tiles_imgs: np.array
    tiles_coords: tuple # coords relative to big image
    lines_coords: List # coords in this stack scope
    scores: torch.tensor = None
    coords: torch.tensor = None


def stack_sniper(ts, imgs_t):
    scores, coords = tsm.forward(ts.tiles_imgs)
    ts.scores = scores
    ts.coords = coords

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



while True:
    for imgs_t, coords in dataloader:
        imgs_t = imgs_t.to(args.device)
        # debug
        img_t = imgs_t[0].cpu()
        img = img_t.squeeze().numpy()
        square = np.zeros_like(img)

        for ts in stack_iterator(n_portions, block_size, imgs_t, coords):
            ts = stack_sniper(ts, imgs_t)

            # loss for TileSniper
            scorings_truth = [torch.tensor(1) if lc else torch.tensor(0) for lc in ts.lines_coords]
            scorings_truth = torch.stack(scorings_truth).unsqueeze(-1).float()
            scorings_truth = scorings_truth.to(args.device)

            # debug
            s = ((float(ts.scores[0].cpu().detach()) + 1) / 2) 
            x1, y1, x2, y2 = ts.tiles_coords
            
            square = cv2.rectangle(square, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (s), -1)

            op = 0.5
            overlay = cv2.addWeighted(img, op, square, 1-op, 0)

            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", overlay)
            cv2.waitKey(1)



            loss = loss_fn(ts.scores, scorings_truth)
            print(s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



       

            
            # walk each tile in stack
            continue
            for s, c, img_t in zip(st.scores, st.coords, imgs_t):
                if not s:
                    continue

                x, y = c

                for _ in range(10):
                    patch = img_t[:, y1:y2, x1:x2]

                    d_score, d_angle = dm.forward()

                    x_patch = x + (patch_magnitude * np.cos(a))
                    y_patch = y + (patch_magnitude * np.sin(a))
                

       

        
        # imshow_sequences

            

        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # x=0















