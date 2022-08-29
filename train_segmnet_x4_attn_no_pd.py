import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms

from datasets import YTB_VOS
from data import processing, sampler, LTRLoader
import data.transforms as dltransforms

from models import create_segmnet_x4_attn_no_pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from util import save_debug, box2roi, box2gauss

if __name__ == '__main__':

    # Train settings
    n_epochs = 16
    batch_size= 16
    print_interval = 1
    normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)
    search_area_factor = 4.0
    output_sz = 224
    center_jitter_factor = {'train': 0, 'test': 1.5}
    scale_jitter_factor = {'train': 0, 'test': 0.25}

    # settings for guassian label
    output_sigma_factor = 1/8
    sigma_factor = output_sigma_factor / search_area_factor # 1/16
    feature_sz = 28 # output size of gaussian map
    kernel_sz = 2 # kernel size of gaussian filter

    model_name = 'segmnet_x4_attn_no_pd'

    temp_path = './temp/' + model_name
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)

    output_path = './train/' + model_name
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    start_epoch = 1
    pretrained_model = './train/%s/model_e00.pth'%model_name

    # Training datasets
    data_root = '/home/yan/Data2/Datasets/ytb-vos/'
    vos_train = YTB_VOS(root=data_root, split='train')

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=normalize_mean, std=normalize_std)])
    data_processing_train = processing.SegmProcessing(search_area_factor=search_area_factor,
                                                      output_sz=output_sz,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='pair',
                                                      transform=transform_train)

    # The sampler for training
    dataset_train = sampler.SegmSampler([vos_train], [1], samples_per_epoch=1000 * batch_size, max_gap=50, processing=data_processing_train)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True, stack_dim=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device : ', device)

    # Create model
    net = create_segmnet_x4_attn_no_pd(pretrained_backbone=True)
    for param in net.backbone.parameters():
        param.requires_grad = False

    # Load pretrained weights
    if pretrained_model is not None:
        net.load_state_dict(torch.load(pretrained_model))
        print('Loading pretrained weights from:', pretrained_model)

    net = net.to(device)
    net.train()

    # Set objective
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        for i, data in enumerate(loader_train):
            data = data.to(device)

            train_images = data['train_images'].permute(1, 0, 2, 3) # [B, 3, H, W]
            train_depths = data['train_depths'].permute(1, 0, 2, 3) # [B, 3, H, W]
            train_masks = data['train_masks'].permute(1, 0, 2, 3)   # [B, 1, H, W]
            train_boxes = data['train_anno'].permute(1, 0)          # [B, 4]
            train_rois = box2roi(train_boxes)                       # [B, 5] -> (batch_idx, x1, y1, x2, y2)
            #
            test_images = data['test_images'].permute(1, 0, 2, 3)
            test_depths = data['test_depths'].permute(1, 0, 2, 3)
            test_boxes = data['test_anno'].permute(1, 0)          # [B, 4]
            test_masks = data['test_masks'].permute(1, 0, 2, 3)   # [B, 1, H, W]
            test_gauss = box2gauss(test_boxes.cpu(), sigma_factor, kernel_sz, feature_sz, output_sz,
                                   end_pad_if_even=False, density=False, uni_bias=0)
            test_gauss = test_gauss.to(device)

            gt_masks = F.interpolate(test_masks, scale_factor=0.5) # [B, 1, 112, 112]
            gt_mask_pair = torch.cat((gt_masks, 1-gt_masks), dim=1).to(device)

            optimizer.zero_grad()
            pred_masks, pred_scores = net(train_images, train_depths, train_masks, train_rois, test_images, test_depths)
            mask_loss = criterion(pred_masks, gt_mask_pair)
            score_loss = torch.mean(torch.sum(torch.sum((pred_scores - test_gauss) ** 2, axis=1), axis=1)) # Bx28x28

            loss = mask_loss + 0.01 * score_loss

            loss.backward()
            optimizer.step()

            print('Epoch: %d, batch: %d, loss: %f, mask loss : %f, score loss : %f'%(epoch, i, loss.item(), mask_loss.item(), score_loss.item()))
            if i % 50 == 0:
                save_debug(temp_path, epoch, i,
                           train_images, train_depths, train_masks, train_boxes,
                           test_images, test_depths, test_masks, test_boxes, test_gauss,
                           pred_masks, pred_scores)

            running_loss += loss.item()
        lr_scheduler.step()
        print('Epoch: %d, loss: %f'%(epoch, running_loss/i))

        torch.save(net.state_dict(), os.path.join(output_path, 'model_e%02d.pth'%epoch))
