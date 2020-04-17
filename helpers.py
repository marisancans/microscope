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

def imshow_sequences(img_rows: List[List], w_name="grid", info_columns = [], render=True, waitKey=0, save_path='', pre_save_callable=None):
    # padding
    max_val = max([len(x) for x in img_rows])

    for idx_row, row in enumerate(img_rows):
        for i in range(max_val - len(row)):
            img_rows[idx_row].append(np.zeros_like(row[-1]))

    if info_columns:
        for c in info_columns:
            if len(c) != len(img_rows):
                print("len(info_column) != len(img_rows)")
                os._exit(0)

            for i, pre in enumerate(c):
                img_rows[i].insert(0, pre)


    stacked_row = [np.hstack(row) for row in img_rows]
    stacked_row_t = [torch.tensor(x).permute(2, 0, 1) for x in stacked_row]
    grid_t = torchvision.utils.make_grid(stacked_row_t, nrow=1)
    grid = grid_t.permute(1, 2, 0).numpy() 
    
    if render:
        cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
        cv2.imshow(w_name, grid)
        cv2.waitKey(waitKey)
    
    if save_path:
        if pre_save_callable:
            grid = pre_save_callable(grid)
        cv2.imwrite(save_path, grid)