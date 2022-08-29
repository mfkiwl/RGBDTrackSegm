import os
import time
import cv2
import math
import random
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize

from models import create_segmnet_x4_attn as create_segmnet
import data.transforms as dltransforms

import data.processing_utils as prutils
from util import visualize_results, mask2box, box2mask, box2roi, box_to_frame, \
                 max_contour_in_mask, write_results_to_file


def read_images(color_path, depth_path, target_depth=5000, bbox=None):
    ''' color: np.array, H*W*3, depth: np.array H*W*3, mask: np.array, H*W '''
    color = cv2.imread(color_path)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # color = np.asarray(color, dtype=np.float32)

    depth = cv2.imread(depth_path, -1)
    if target_depth is None and bbox is not None:
        x0, y0, w, h = bbox
        target_region = np.asarray(depth[int(y0):int(y0+h), int(x0):int(x0+w)], dtype=np.float32)
        target_region = np.nan_to_num(target_region, -1)
        target_depth = np.median(target_region[target_region > 0])
        if target_depth < 50:
            target_depth = 2000

    if np.max(depth) > 10:
        depth = np.asarray(depth, dtype=np.float32) / (target_depth*2)
        depth = np.clip(depth, 0.0, 1.0)
        depth = 1 - depth # reverse depth ???
    depth = cv2.applyColorMap(np.asarray(depth*255, dtype=np.uint8), cv2.COLORMAP_JET)
    # depth = np.asarray(depth, dtype=np.float32)

    return color, depth, target_depth

def process_images(device, color, depth, mask, bbox, transform_func, area_factor=4.0, output_sz=224):
    '''Return : crop_color, torch Tensor, 1*3*224*224,
                crop_depth, torch Tensor, 1*3*224*224,
                crop_mask,  torch Tensor, 1*1*224*224,
                crop_bbox,  torch Tensor, 1*4, (x0, y0, w, h)
                rois,       torch Tensor, 1*5, (batch_idx, x0, y0, x1, y1)
                scale_factor, float, the scale factor in crop and resize
    '''
    # stack, Only one instance
    color, depth, mask, bbox = [color], [depth], [mask], torch.tensor([bbox])

    # crop and resize
    crop_color, crop_bbox, scale_factor = prutils.centered_crop(color, bbox, area_factor, output_sz)
    crop_depth, _, _ = prutils.centered_crop(depth, bbox, area_factor, output_sz, pad_val=0)

    # ToTensor and Normalize
    crop_color = transform_func(crop_color[0]) # torch Tensor, 3*224*224, [0, 1.0]
    crop_depth = transform_func(crop_depth[0]) # torch Tensor, 3*224*224

    crop_mask = None
    if mask[0] is not None:
        crop_mask, _, _ = prutils.centered_crop(mask, bbox, area_factor, output_sz)
        crop_mask = torch.from_numpy(np.expand_dims(np.asarray(crop_mask[0], dtype=np.float32), axis=0)) # torch Tensor, 1*224*224
        crop_mask = crop_mask.unsqueeze(0).to(device)

    crop_color = crop_color.unsqueeze(0).to(device)
    crop_depth = crop_depth.unsqueeze(0).to(device)
    crop_bbox = crop_bbox[0].unsqueeze(0).to(device) # 1,4
    rois = box2roi(crop_bbox) # Torch tensor, 1*5
    scale_factor = scale_factor[0]

    return crop_color, crop_depth, crop_mask, crop_bbox, rois, scale_factor


def create_tracker(device, pretrained_backbone=False, pretrained_model=None):
    tracker = create_segmnet(pretrained_backbone=pretrained_backbone)
    if pretrained_model is not None:
        tracker.load_state_dict(torch.load(pretrained_model))
        print('Loading pretrained weights from:', pretrained_model)
    tracker = tracker.to(device)
    return tracker

def init_tracker(tracker, train_color, train_depth, train_mask, train_rois):
    train_rgbd_feat = tracker.extract_backbone_feat(train_color, train_depth)
    train_rgbd_feat = tracker.pixel_decoder(train_rgbd_feat)

    # train_pixel_feat = tracker.flatten_pixel_features(train_rgbd_feat)

    train_rgbd_feat = [f for f in train_rgbd_feat.values()]
    fg_query, bg_query = tracker.init_queries(train_rgbd_feat[:4], train_mask, train_rois)

    # fg_query, _ = tracker.query_decoder(fg_query, train_pixel_feat[:4])
    # bg_query, _ = tracker.query_decoder(bg_query, train_pixel_feat[:4])

    return fg_query, bg_query

def track(tracker, test_color, test_depth, fg_query, bg_query):
    test_rgbd_feat = tracker.extract_backbone_feat(test_color, test_depth)
    test_rgbd_feat = tracker.pixel_decoder(test_rgbd_feat)
    # Update queries
    test_pixel_feat = tracker.flatten_pixel_features(test_rgbd_feat)

    updated_fg_query, score = tracker.query_decoder(fg_query, test_pixel_feat[:4])
    updated_bg_query, _ = tracker.query_decoder(bg_query, test_pixel_feat[:4])

    # Tracking
    mask = tracker.mask_head(test_rgbd_feat['sep'], updated_fg_query, updated_bg_query, score) # [1, 2, 112, 112]
    mask = F.interpolate(mask, scale_factor=2)                                # [1, 2, 224, 224]
    mask = F.softmax(mask, dim=1)                                             # [1, 2, 224, 224]
    mask = mask.clone().detach().cpu().numpy()[0, 0, :, :] # 224*224
    mask = (mask > 0.5).astype(np.uint8)
    mask = max_contour_in_mask(mask) # use the largest contour
    bbox = mask2box(mask) # box in 224*224

    score = score.clone().detach().cpu().numpy()[0, 0, ...]

    return mask, bbox, score


if __name__ == '__main__':

    # Settings
    visualize = False
    save_mask = False
    search_area_factor = 4.0
    output_sz = 224
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    # transforms = torchvision.transforms.Compose([ToTensor(), Normalize(mean=normalize_mean, std=normalize_std)])
    transforms = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                 torchvision.transforms.Normalize(mean=normalize_mean, std=normalize_std)])

    model_name = 'segmnet_x4_attn'
    model_path = './train/%s/model_e15.pth'%model_name
    dataset_name = 'CDTB'
    data_root = '/home/yan/Data4/Datasets/%s/sequences/'%dataset_name

    output_path = os.path.join('./test/', model_name, dataset_name, 'rgbd-unsupervised')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    sequences = os.listdir(data_root)
    try:
        sequences.remove('list.txt')
    except:
        pass

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device : ', device)

    tracker = create_tracker(device, pretrained_backbone=False, pretrained_model=model_path)

    # random_seq = random.randint(0, len(sequences))

    for seq in sequences:
        print(seq)
        color_path = os.path.join(data_root, seq, 'color')
        depth_path = os.path.join(data_root, seq, 'depth')
        gt_path = os.path.join(data_root, seq, 'groundtruth.txt')

        num_frames = len(os.listdir(color_path))

        prediction_bbox = []
        prediction_conf = []
        prediction_time = []

        with open(gt_path, 'r') as fp:
            gt_bboxes = fp.readlines()
        gt_bboxes = [line.strip() for line in gt_bboxes]
        gt_bboxes = np.asarray([[int(float(bb)) if bb!='nan' else 0 for bb in box.split(',')] for box in gt_bboxes])

        out_result_path = os.path.join(output_path, seq)
        os.makedirs(out_result_path, exist_ok=True)

        if save_mask:
            out_mask_path = os.path.join(out_result_path, 'mask')
            os.makedirs(out_mask_path, exist_ok=True)

        # Initial tracker
        tic = time.time()
        ii = 1
        train_bbox = gt_bboxes[0]
        train_color, train_depth, target_depth = read_images(os.path.join(color_path, '%08d.jpg'%ii),
                                                             os.path.join(depth_path, '%08d.png'%ii),
                                                             target_depth=None, bbox=train_bbox)
        print('target depth : ', target_depth)
        train_mask = box2mask(train_bbox, train_color.shape[0], train_color.shape[1]) # [H, W]
        # crop color, crop depth, crop mask, crop bbox, scale factor
        train_cc, train_cd, train_cm, train_cb, train_rois, train_sf = process_images(device, train_color, train_depth, train_mask, train_bbox,
                                                                                      transforms, area_factor=search_area_factor, output_sz=output_sz)
        # Initial tracker
        init_fg_query, init_bg_query = init_tracker(tracker, train_cc, train_cd, train_cm, train_rois)

        toc = time.time()
        prediction_time.append(toc-tic)
        prediction_bbox.append(1)
        prediction_conf.append(1)

        if visualize:
            fig, axes = plt.subplots(3, 2, figsize=(6, 6))
            # visualize_results(fig, axes, ii,
            #                   train_color, train_depth, train_bbox, train_cc, train_cd,
            #                   train_cm, train_bbox, train_cm, train_cb, use_norm=True)

        # Track
        total_time = 0
        prev_bbox = train_bbox
        for ii in range(2, num_frames+1):

            gt_bbox = gt_bboxes[ii-1]

            test_color, test_depth, _ = read_images(os.path.join(color_path, '%08d.jpg'%ii),
                                                    os.path.join(depth_path, '%08d.png'%ii),
                                                    target_depth=target_depth)
            test_cc, test_cd, _, _, _, test_cf = process_images(device, test_color, test_depth, None, prev_bbox,
                                                                transforms, area_factor=search_area_factor, output_sz=output_sz)

            tic = time.time()
            #
            pred_mask, pred_bbox, pred_prob = track(tracker, test_cc, test_cd, init_fg_query, init_bg_query)
            # map bbox to image coordinates
            pred_bbox_in_frame = box_to_frame(pred_bbox, prev_bbox, test_cf, search_area_factor)

            toc = time.time()
            prediction_time.append(toc-tic)
            prediction_bbox.append(pred_bbox_in_frame)
            prediction_conf.append(np.max(pred_prob))

            total_time += toc - tic
            # update search region
            new_target_sz = pred_bbox_in_frame[2] * pred_bbox_in_frame[3]
            if new_target_sz >= 0.75*train_bbox[2]*train_bbox[3]:
                prev_bbox = pred_bbox_in_frame

            if visualize:
                visualize_results(fig, axes, ii,
                                  np.asarray(test_color, dtype=np.uint8), np.asarray(test_depth, dtype=np.uint8), gt_bbox, test_cc, test_cd,
                                  pred_mask, pred_bbox_in_frame, pred_prob, pred_bbox, use_norm=True)


        # # output
        write_results_to_file(prediction_bbox, prediction_conf, prediction_time, out_result_path, seq)

        total_time = total_time / (num_frames-1)
        print('average time per frame : ', total_time, ' speed : ', 1/total_time)

        if visualize:
            plt.close()
