import cv2
import numpy as np
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass, field

from unet_model import UNet
from reseg import ReSeg

from datasets import SynthDataset, collate_fn
from discriminative import DiscriminativeLoss
from dice import dice_loss_fn
import hdbscan
from sklearn.decomposition import PCA
from cluster import Clusterer
import argparse
from helpers import RangeNormalize

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--debug', type=arg_to_bool, default=False)
parser.add_argument('--segment_channels', type=int, default=16)
args = parser.parse_args()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




# params
img_size = 512
n_portions = 10
block_size = img_size // n_portions # 50
batch_size = 4
layer_count = 10
n_objects_max = layer_count

#discriminative
delta_var = 0.5
delta_dist = 1.5

patch_size = 50

seg_binary_thershold = 0.5


transforms=transforms.Compose([
    AddGaussianNoise(0.1, 0.025)
])

dataset = SynthDataset(img_size=img_size, transforms=transforms, batch_size=batch_size, layer_count=layer_count)
dataloader = DataLoader(batch_size=args.batch_size, dataset=dataset, num_workers=0, collate_fn=collate_fn)

n_classes = 2

# instance_model = ReSeg(n_classes=n_classes, use_instance_seg=True, pretrained=False, usegpu=True).to(args.device)
segmenter_model = UNet(n_channels=1, n_classes=2).to(args.device)
instance_model = UNet(n_channels=1, n_classes=6).to(args.device)

loss_binary = torch.nn.BCEWithLogitsLoss()
discriminative_loss = DiscriminativeLoss(delta_var=delta_var, delta_dist=delta_dist, norm=2, usegpu=True)
cross_entropy_fn = torch.nn.CrossEntropyLoss() 

optimizer = torch.optim.Adam(list(instance_model.parameters()) + list(segmenter_model.parameters()), 0.001)


def stack_iterator(n_portions, block_size, stacks = []):
    for y in range(n_portions):
        for x in range(n_portions):
            y1 = y * block_size
            x1 = x * block_size
            y2 = (y + 1) * block_size
            x2 = (x + 1) * block_size

            crops = [s[:, :, y1:y2, x1:x2] for s in stacks]

            yield crops, (x1, y1, x2, y2)

# colormaps: https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python
def show(name, img, waitkey=0, color_map = None, pre_lambda=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if pre_lambda:
            img = pre_lambda(img) 

    if color_map:
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(img, color_map)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(waitkey)


def debug(debug_pack, debug_canvas, coords):
    for (b_inputs, b_istances, b_masks, b_segmented, cluster_result, idxs), debug_canv in zip(debug_pack, debug_canvas):
        if not cluster_result.success:
            print("Skipped, not enough embeddings")
            continue

        # draw reduced 
        plt.cla()
        plt.scatter(cluster_result.reduced.T[0], cluster_result.reduced.T[1], c = cluster_result.colors, alpha=1.0, linewidths=0, s=60)

        for i, (d, l) in enumerate(zip(cluster_result.reduced, cluster_result.labels)):
            x, y = d
            # plt.text(x, y, f'{l}', fontsize=9)
            if l == '-1':
                plt.scatter(x, y, c='red', alpha=0.5, linewidths=0, marker='X', s=90)

        plt.show()

        # remapping to spatial dim
        remap_canvas = np.zeros((block_size, block_size, 3))
        
        for n, c in zip(idxs, cluster_result.colors):
            x, y = n
            remap_canvas[x][y] = c

        # overlay to canvas
        x1, y1, x2, y2 = coords
        debug_canv.remapped[y1:y2, x1:x2] = remap_canvas
        canvas_remapped = np.vstack([x.remapped for x in debug_canvas])
        # b_inputs = b_inputs.reshape(-1, block_size)
        
        show("canvas_remapped", canvas_remapped, 1)


        b_inputs = b_inputs.reshape(-1, block_size)
        b_istances = b_istances.reshape(-1, block_size)
        b_masks = b_masks.contiguous().view(-1, block_size)
        b_segmented = b_segmented.view(-1, block_size)

        rn = RangeNormalize(0, 1.0)

        show("inputs", b_inputs, 1)
        show("instances", rn(b_istances), 1, cv2.COLORMAP_HSV )
        show("masks", b_masks, 1)
        show("segmeted", b_segmented, 1)
        show("remapped", remap_canvas, 1)
        cv2.waitKey(1)

class DebugCanvas:
    def __init__(self, size):
        self.inputs: np.array = np.zeros((size, size))
        self.instances: np.array = np.zeros((size, size))
        self.masks: np.array = np.zeros((size, size))
        self.segmented: np.array = np.zeros((size, size))
        self.remapped: np.array = np.zeros((size, size, 3))

st_args = [n_portions, block_size]

normalize = lambda x : (x - x.min())/(x.max())


# cluster
clustering_algo = hdbscan.HDBSCAN(
    min_cluster_size=32,
    min_samples=32, 
    alpha=1.0, # do not change
    algorithm='generic', # for stability generic
    # metric='cosine', # cosine without PCA whitening
    metric='cosine', # PCA whitening + euclidean works good
    core_dist_n_jobs=8,
    match_reference_implementation=True,
    allow_single_cluster=True # some cases work better others not, need to test
)

# PCA
pca_n_components = 2

pca = PCA(
    n_components=pca_n_components, # dim to reduce it
    whiten=False, # if cosine then whiten not needed
    svd_solver='full'
)       




batch = -1
while True:
    batch += 1
    for layers, combined_imgs_t, fore_back_masks_t, masks_t in dataloader:
        combined_imgs_t = combined_imgs_t.to(args.device)
        fore_back_masks_t = fore_back_masks_t.to(args.device)
        masks_t = masks_t.to(args.device)

        # debug
        debug_canvas = [DebugCanvas(img_size) for _ in range(batch_size)]

        pack = stack_iterator(*st_args, stacks=[combined_imgs_t, fore_back_masks_t, masks_t])

        for (c_combined, c_fore_back, c_masks), coords in pack:

            ins_seg_predictions = instance_model.forward(c_combined)
            sem_seg_predictions = segmenter_model.forward(c_combined)

            # mask instance segmentations with semantic segmetnation
            masked_np = []
            nonzeros = []
            nonzeros_idxs = []

            for ins, seg in zip(ins_seg_predictions, sem_seg_predictions):
                foreground, background = seg

                thresh_foreground = foreground > seg_binary_thershold  
                thresh_background = background > seg_binary_thershold  

                # show("thresh", thresh_foreground, 0, lambda x: x*255.0)
                nonzeros_idxs.append(thresh_foreground.nonzero())         

                non = [torch.masked_select(i, thresh_foreground) for i in ins]
                non = torch.stack(non)
                non = non.permute(1, 0) # swap embeds to piexel dims
                nonzeros.append(non)


            
            # determine mask indicies per stack
            n_obj_idxs = []

            for c_mask in c_masks:
                n = []
                for i, x in enumerate(c_mask):
                    if x.max():
                        n.append(i)
                n_obj_idxs.append(n)      

            # resort masks, so first ones are non-empty
            c_masks_sorted = []
            for n_idx, c_mask in zip(n_obj_idxs, c_masks):
                cm = []
                cm_empty = []
                for i, _ in enumerate(c_mask):
                    if i in n_idx:
                        cm.append(c_mask[i])
                    else:
                        cm_empty.append(c_mask[i])
                cm.extend(cm_empty)
                c_masks_sorted.append(torch.stack(cm))

            c_masks_sorted = torch.stack(c_masks_sorted)

            # pick non-empty layer batches
            c_masks_non_empty = []
            ins_seg_predictions_non_empty = []
            n_obj_idxs_non_empty = []

            for c_mask, ins, n_obj in zip(c_masks_sorted, ins_seg_predictions, n_obj_idxs):
                if len(n_obj):
                    c_masks_non_empty.append(c_mask)
                    ins_seg_predictions_non_empty.append(ins)
                    n_obj_idxs_non_empty.append(len(n_obj))

            disc_loss = 0

            if n_obj_idxs_non_empty:
                c_masks_non_empty = torch.stack(c_masks_non_empty)
                ins_seg_predictions_non_empty = torch.stack(ins_seg_predictions_non_empty)


                n_objects = torch.tensor(n_obj_idxs_non_empty).to(args.device)
                disc_loss = discriminative_loss.forward(ins_seg_predictions_non_empty, c_masks_non_empty, n_obj_idxs_non_empty, n_objects_max)


            # cross entropy loss
            # _, sem_seg_annotations_criterion_ce = sem_seg_predictions.max(1)
            # ce_loss = cross_entropy_fn(sem_seg_predictions.permute(0, 2, 3, 1).contiguous().view(-1, n_classes), sem_seg_annotations_criterion_ce.view(-1))
            # loss += ce_loss

            # dice loss
            dice_loss = dice_loss_fn(sem_seg_predictions, c_fore_back)

            print("DISC LOSS", disc_loss)
            loss = disc_loss + dice_loss 
        
            embedding_clusterer = Clusterer(clustering_algo, pca_n_components, debug=True) 

            nonzeros_np = [x.detach().cpu().numpy() for x in nonzeros]
            nonzeros_idxs_np = [x.detach().cpu().numpy() for x in nonzeros_idxs]
            cluster_results = [embedding_clusterer(n, pca) for n in nonzeros_np]
            
            debug_pack = zip(c_combined, ins_seg_predictions, c_masks, sem_seg_predictions, cluster_results, nonzeros_idxs_np)
            debug(debug_pack, debug_canvas, coords)

            

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)












