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
        layers, combined = self.sg.gen_img(2, self.start_pos)

        imgs_t = torch.tensor(combined).float()
        imgs_t = imgs_t.unsqueeze(0) # Add channel dimennsion
        return layers, imgs_t

def collate_fn(sub_batches):
    b_layers = []
    b_imgs = []
    
    for layers, imgs_t in sub_batches:
        b_imgs.append(imgs_t)
        b_layers.append(layers) 
    
    return torch.stack(b_imgs), b_layers