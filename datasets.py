from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from typing import Tuple
from line_gen import SquiggleGen
import torch
import numpy as np

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