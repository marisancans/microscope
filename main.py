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
img_size = 32
n_portions = 10
block_size = img_size // n_portions # 50

patch_magnitude = 25.0
patch_size = 50

seg_binary_thershold = 0.5


transforms=transforms.Compose([
    AddGaussianNoise(0.01, 0.01)
])

dataset = SynthDataset(img_size=img_size, transforms=transforms)
dataloader = DataLoader(batch_size=args.batch_size, dataset=dataset, num_workers=0, collate_fn=collate_fn)

n_classes = 2

# instance_model = ReSeg(n_classes=n_classes, use_instance_seg=True, pretrained=False, usegpu=True).to(args.device)
segmenter_model = UNet(n_channels=1, n_classes=2).to(args.device)
instance_model = UNet(n_channels=1, n_classes=6).to(args.device)

loss_binary = torch.nn.BCEWithLogitsLoss()
discriminative_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2, usegpu=True)
cross_entropy_fn = torch.nn.CrossEntropyLoss() 

optimizer = torch.optim.Adam(list(instance_model.parameters()) + list(segmenter_model.parameters()), 0.001)


def stack_iterator(n_portions, block_size, imgs_t):
    for y in range(n_portions):
        for x in range(n_portions):
            y1 = y * block_size
            x1 = x * block_size
            y2 = (y + 1) * block_size
            x2 = (x + 1) * block_size
            tile_imgs = imgs_t[:, :, y1:y2, x1:x2]

            yield tile_imgs, (x1, y1, x2, y2)

def show(name, img, waitkey=0, pre_lambda=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if pre_lambda:
            img = pre_lambda(img) 

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(1)

st_args = [n_portions, block_size]

normalize = lambda x : (x - x.min())/(x.max() - x.min())


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

        # pack = zip(stack_iterator(*st_args, combined_imgs_t), stack_iterator(*st_args, combined_masks_t), stack_iterator(*st_args, masks_t))

        # for (ts, ts_coords), (ts_combined_mask, ts_combined_mask_coords), (ts_mask, ts_mask_coords) in pack:

        # masks_pred = segmenter_model.forward(combined_imgs_t)
        # sem_seg_predictions, ins_seg_predictions = instance_model.forward(combined_imgs_t)
        ins_seg_predictions = instance_model.forward(combined_imgs_t)
        sem_seg_predictions = segmenter_model.forward(combined_imgs_t)
        # x=0

        # p = masks_pred[0][0].cpu().detach().numpy()
        # cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
        # cv2.imshow("pred", p)

        # m = ts_combined_mask[0][0].cpu().detach().numpy().astype(float)
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

            show("thresh", thresh_foreground, 0, lambda x: x*255.0)
            show("foreground", foreground, 0)
            show("background", background, 0)
            nonzeros_idxs.append(thresh_foreground.nonzero())         

            # sub = [] 
            non = [torch.masked_select(i, thresh_foreground) for i in ins]
            non = torch.stack(non)
            non = non.permute(1, 0) # swap embeds to piexel dims
            nonzeros.append(non)

            for i in ins:
                i[thresh_background] = 0
                # sub.append(i)

            # sub = torch.cat(sub, 0)          

            # show("w_mask", normalize(sub.detach().cpu().numpy()))
            # show("sub", s)
        # nonzeros = [x.view(-1) for x in nonzeros]
        # nonzeros = [x.detach().cpu().numpy() for x in nonzeros]
        x=0



        n_objects = torch.tensor([2]* 6).to(args.device)

        # discriminative loss
        loss = discriminative_loss.forward(ins_seg_predictions, masks_t, n_objects, 2)

        # cross entropy loss
        # _, sem_seg_annotations_criterion_ce = sem_seg_predictions.max(1)
        # ce_loss = cross_entropy_fn(sem_seg_predictions.permute(0, 2, 3, 1).contiguous().view(-1, n_classes), sem_seg_annotations_criterion_ce.view(-1))
        # loss += ce_loss

        # dice loss
        # sem_seg_predictions, gpu_sem_seg_annotations
        dice_loss = dice_loss_fn(sem_seg_predictions, fore_back_masks_t)
        loss += dice_loss

        ins_seg_np = ins_seg_predictions.detach().cpu().numpy()
        ins_seg_np = normalize(ins_seg_np)

        sem_seg_np = sem_seg_predictions.detach().cpu().numpy()

        masks_np = masks_t.detach().cpu().numpy()




        

        if batch > 10:
            embedding_clusterer = Clusterer(clustering_algo, pca_n_components, debug=True) 

            # per sample in batch
            # for n in nonzeros:
            # non_np = [x.detach().cpu().numpy() for x in n]
            labels, mvs, colors = embedding_clusterer(nonzeros[0].detach().cpu().numpy(), pca)

            nonzeros_idxs_np = [x.detach().cpu().numpy() for x in nonzeros_idxs]

            n1 = nonzeros_idxs_np[0]

            remapped = np.zeros((img_size, img_size, 3))

            for n, c in zip(n1, colors):
                x, y = n
                remapped[x][y] = c

            show("remapped", remapped, 1)



            for b_istances, b_masks, b_segmented in zip(ins_seg_np, masks_np, sem_seg_np):
            #     # print(input_np.shape, target_np.shape)
            #     # b = b.permute(1, 2, 0).numpy()
                b_istances = np.vstack(b_istances)
            #     b_masks = np.vstack(b_masks)
            #     b_segmented = np.vstack(b_segmented)

                cv2.namedWindow("instances", cv2.WINDOW_NORMAL)
                cv2.imshow("instances", b_istances)
            #     cv2.namedWindow("masks", cv2.WINDOW_NORMAL)
            #     cv2.imshow("masks", b_masks)
            #     cv2.namedWindow("segmented", cv2.WINDOW_NORMAL)
            #     cv2.imshow("segmeted", b_segmented)
                cv2.waitKey(1)

    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)












