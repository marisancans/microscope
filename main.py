import cv2
import numpy as np

# from unet import UNet

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from dataclasses import dataclass
from typing import Tuple
from line_gen import SquiggleGen

@dataclass
class SynthDataset(Dataset):
    batch_size: int = 64
    img_size: int = 1000
    start_pos: Tuple[int, int] = (500, 500)

    def __post_init__(self):
        self.sg = SquiggleGen(size = self.img_size, debug=False)

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        imgs, coords = self.sg.gen_imgs(self.batch_size, self.start_pos)
        # offset

        # imgs_t = torch.tensor(imgs)
        # img_t = img_t.permute(2, 0, 1) # (H, W, C) --> (C, H, W)    
        return imgs, coords

def collate_fn(sub_batches):
    b_imgs = []
    b_coords = []
    
    for imgs, coords in sub_batches:
        b_imgs.extend(imgs)
        b_coords.extend(coords) 
    
    return np.array(b_imgs), np.array(b_coords)



# params
img_size = 1000
n_portions = 10
block_size = img_size // n_portions

dataset = SynthDataset(batch_size=4, img_size=img_size)
dataloader = DataLoader(dataset=dataset, num_workers=0, collate_fn=collate_fn)





# # cv2.imshow('img', img)
# # cv2.waitKey(1)


# # img = np.zeros((400, 300), dtype=np.uint8)
# orignal = cv2.imread("pic.png")
# orignal = cv2.cvtColor(orignal, cv2.COLOR_BGR2GRAY)


while True:
    for imgs, coords in dataloader:
        # split each img into grid

        for img in imgs:
            for y in range(n_portions):
                for x in range(n_portions):
                    y1 = y * block_size
                    x1 = x * block_size
                    y2 = (y + 1) * block_size
                    x2 = (x + 1) * block_size
                    tiles = img[y1:y2,x1:x2]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (1))
            cv2.imshow("img", img)
            cv2.waitKey(0)
            x=0















