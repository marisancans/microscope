import cv2
import numpy as np
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

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
img_size = 256
n_portions = 10
block_size = img_size // n_portions # 50
batch_size = 6
n_objects_max = 2

#discriminative
delta_var = 0.5
delta_dist = 1.5

patch_magnitude = 25.0
patch_size = 50

seg_binary_thershold = 0.5


transforms=transforms.Compose([
    AddGaussianNoise(0.2, 0.1)
])

dataset = SynthDataset(img_size=img_size, transforms=transforms, batch_size=batch_size)
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

def show(name, img, waitkey=0, pre_lambda=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if pre_lambda:
            img = pre_lambda(img) 

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(1)

st_args = [n_portions, block_size]

normalize = lambda x : (x - x.min())/(x.max())


# cluster
clustering_algo = hdbscan.HDBSCAN(
    min_cluster_size=64,
    min_samples=128, 
    alpha=1.0, # do not change
    algorithm='generic', # for stability generic
    # metric='cosine', # cosine without PCA whitening
    metric='cosine', # PCA whitening + euclidean works good
    core_dist_n_jobs=8,
    match_reference_implementation=True,
    allow_single_cluster=False # some cases work better others not, need to test
)

# PCA
pca_n_components = 2

pca = PCA(
    n_components=pca_n_components, # dim to reduce it
    whiten=True, # if cosine then whiten not needed
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
        canvas_truth = np.zeros((img_size, img_size))
        canvas_pred = np.zeros((img_size, img_size))
        canvas_inst = np.zeros((img_size, img_size))

        # pack = stack_iterator(*st_args, stacks=[combined_imgs_t, fore_back_masks_t, masks_t])

        # for (c_combined, c_fore_back, c_masks), coords in pack:

        ins_seg_predictions = instance_model.forward(combined_imgs_t)
        sem_seg_predictions = segmenter_model.forward(combined_imgs_t)


        # p = ins_seg_predictions[0][0].cpu().detach().numpy()
        # cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
        # cv2.imshow("pred", p)

        # m = sem_seg_predictions[0][0].cpu().detach().numpy().astype(float)
        # cv2.namedWindow("truth", cv2.WINDOW_NORMAL)
        # cv2.imshow("truth", m)

        # mm = instance_pred[0][0].cpu().detach().numpy().astype(float)

        # x1, y1, x2, y2 = ts_coords
        # canvas_truth[y1:y2, x1:x2] = p

        # x1, y1, x2, y2 = ts_combined_mask_coords
        # canvas_pred[y1:y2, x1:x2] = m

        # x1, y1, x2, y2 = ts_mask_coords
        # canvas_inst[y1:y2, x1:x2] = mm

        # cv2.namedWindow("pred_tiles", cv2.WINDOW_NORMAL)
        # cv2.imshow("pred_tiles", canvas_truth)

        # cv2.namedWindow("truth_tiles", cv2.WINDOW_NORMAL)
        # cv2.imshow("truth_tiles", canvas_pred)

        # cv2.namedWindow("instance", cv2.WINDOW_NORMAL)
        # cv2.imshow("instance", canvas_inst)


        # cv2.waitKey(1)


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

            

        n_objects = [n_objects_max] * batch_size
        n_objects = torch.tensor(n_objects).to(args.device)

        # discriminative loss
        loss = discriminative_loss.forward(ins_seg_predictions, masks_t, n_objects, n_objects_max)

        # cross entropy loss
        # _, sem_seg_annotations_criterion_ce = sem_seg_predictions.max(1)
        # ce_loss = cross_entropy_fn(sem_seg_predictions.permute(0, 2, 3, 1).contiguous().view(-1, n_classes), sem_seg_annotations_criterion_ce.view(-1))
        # loss += ce_loss

        # dice loss
        dice_loss = dice_loss_fn(sem_seg_predictions, fore_back_masks_t)
        loss += dice_loss
       

        if batch > 10:
            embedding_clusterer = Clusterer(clustering_algo, pca_n_components, debug=True) 


            nonzeros_np = [x.detach().cpu().numpy() for x in nonzeros]
            nonzeros_idxs_np = [x.detach().cpu().numpy() for x in nonzeros_idxs]
            cluster_results = [embedding_clusterer(n, pca) for n in nonzeros_np]
          
            debug_pack = zip(combined_imgs_t, ins_seg_predictions, masks_t, sem_seg_predictions, cluster_results, nonzeros_idxs_np)

            for b_inputs, b_istances, b_masks, b_segmented, (labels, mvs, colors, reduced), idxs in debug_pack:

                # draw reduced 
                plt.cla()
                plt.scatter(reduced.T[0], reduced.T[1], c = colors, alpha=1.0, linewidths=0, s=60)

                for i, (d, l) in enumerate(zip(reduced, labels)):
                    x, y = d
                    # plt.text(x, y, f'{l}', fontsize=9)
                    if l == '-1':
                        plt.scatter(x, y, c='red', alpha=0.5, linewidths=0, marker='X', s=90)

                plt.show()

                # remapping to spatial dim
                remap_canvas = np.zeros((img_size, img_size, 3))
                
                for n, c in zip(idxs, colors):
                    x, y = n
                    remap_canvas[x][y] = c

                b_inputs = b_inputs.reshape(-1, img_size)
                b_istances = b_istances.reshape(-1, img_size)
                b_masks = b_masks.view(-1, img_size)
                b_segmented = b_segmented.view(-1, img_size)

                rn = RangeNormalize(0, 1.0)

                show("inputs", b_inputs)
                show("instances", rn(b_istances))
                show("masks", b_masks)
                show("segmeted", b_segmented)
                show("remapped", remap_canvas)
                cv2.waitKey(1)

    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)












