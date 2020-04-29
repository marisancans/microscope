from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from typing import Tuple
from line_gen import SquiggleGen
import torch
import numpy as np
import cv2

@dataclass
class SynthDataset(Dataset):
    img_size: int = 500
    start_pos: Tuple[int, int] = None
    transforms: list = None

    def __post_init__(self):
        if not self.start_pos:
            self.start_pos = (self.img_size // 2, self.img_size // 2)
        self.sg = SquiggleGen(size = self.img_size, debug=False)

    def __len__(self):
        return 6


    def __getitem__(self, idx):
        layers, combined_img, foreground_mask = self.sg.gen_img(2, self.start_pos)

        combined_img_t = torch.tensor(combined_img).float()
        combined_img_t = combined_img_t.unsqueeze(0) # Add channel dimennsion

        masks_t = [torch.tensor(l.mask).float() for l in layers]
        masks_t = torch.stack(masks_t) # Add channel dimennsion

        background_mask = cv2.bitwise_not(foreground_mask.astype(np.uint8)) - 254
        background_mask_t = torch.tensor(background_mask).float()
        background_mask_t = background_mask_t.unsqueeze(0) # Add channel dimennsion

        foreground_mask_t = torch.tensor(foreground_mask).float()
        foreground_mask_t = foreground_mask_t.unsqueeze(0) # Add channel dimennsion

        fore_back_mask_t = torch.cat([foreground_mask_t, background_mask_t])

        if self.transforms:
            combined_img_t = self.transforms(combined_img_t)

        return layers, combined_img_t, fore_back_mask_t, masks_t

def collate_fn(sub_batches):
    b_layers = []
    b_combined_imgs = []
    b_fore_back_masks= []
    b_masks = []
    
    for layers, combined_img_t, fore_back_mask_t, masks_t in sub_batches:
        b_layers.append(layers) 
        b_combined_imgs.append(combined_img_t)
        b_fore_back_masks.append(fore_back_mask_t)
        b_masks.append(masks_t)
    
    return b_layers, torch.stack(b_combined_imgs), torch.stack(b_fore_back_masks), torch.stack(b_masks)