import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import cv2

def tensor2numpy(tensor, batch_idx=0):
    if tensor is not None and torch.is_tensor(tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 3, 1) # BxHxWxC
        tensor = tensor.clone().detach().cpu().numpy()[batch_idx, ...]
    return tensor

def restore_from_norm(img, normalize_mean = [0.485, 0.456, 0.406], normalize_std = [0.229, 0.224, 0.225]):
    img = img * normalize_std + normalize_mean
    img = np.clip(img, 0, 1)
    return img

def imshow(ax, im, title=''):
    ax.cla()
    ax.axis('off')
    if im is not None:
        ax.imshow(im)
        ax.set_title(title)

def add_rect(ax, bbox, color='r'):
    if bbox is not None:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                  linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

def visualize_results(fig, axes, frame_id,
                      color, depth, gt_bbox, search_color, search_depth,
                      pred_mask, pred_bbox, pred_prob, pred_bbox_in_search,
                      use_norm=True):

    if fig is None and len(axes) == 0:
        fig, axes = plt.subplots(2, 4, figsize=(6, 6))

    color = tensor2numpy(color)
    depth = tensor2numpy(depth)
    gt_bbox = tensor2numpy(gt_bbox) # (4,)

    search_color = tensor2numpy(search_color)
    search_depth = tensor2numpy(search_depth)
    pred_prob = tensor2numpy(pred_prob) # HxWx1
    pred_mask = tensor2numpy(pred_mask) # HxWx1
    pred_bbox = tensor2numpy(pred_bbox) # (4,)
    pred_bbox_in_search = tensor2numpy(pred_bbox_in_search) # (4,)

    if use_norm:
        # color = restore_from_norm(color)
        # depth = restore_from_norm(depth)
        search_color = restore_from_norm(search_color)
        search_color = np.clip(search_color, 0.0, 1.0)
        search_depth = restore_from_norm(search_depth)
        search_depth = np.clip(search_depth, 0.0, 1.0)

    plt.cla()
    imshow(axes[0, 0], color, title='Frame %d, Color'%frame_id)
    imshow(axes[0, 1], depth, title='Depth')

    imshow(axes[1, 0], search_color, title='search color')
    imshow(axes[1, 1], search_depth, title='search depth')

    imshow(axes[2, 0], pred_mask, title='pred mask')
    imshow(axes[2, 1], pred_prob, title='pred prob : %f'%np.max(pred_prob))

    add_rect(axes[0, 0], gt_bbox, color='g')
    add_rect(axes[0, 0], pred_bbox, color='r')
    add_rect(axes[1, 0], pred_bbox_in_search, color='r')

    plt.pause(0.0001)
    plt.show(block=False)

def save_debug(temp_path, epoch, batch,
               train_images, train_depths, train_masks, train_boxes,
               test_images, test_depths, test_masks, test_boxes, test_gauss,
               prediction_mask, scores):

    train_rgb = tensor2numpy(train_images)
    train_rgb = restore_from_norm(train_rgb)
    train_d = tensor2numpy(train_depths)
    train_d = restore_from_norm(train_d)

    train_m = tensor2numpy(train_masks)
    train_b = tensor2numpy(train_boxes)

    test_rgb = tensor2numpy(test_images)
    test_rgb = restore_from_norm(test_rgb)

    test_d = tensor2numpy(test_depths)
    test_d = restore_from_norm(test_d)

    test_m = tensor2numpy(test_masks)
    test_b = tensor2numpy(test_boxes)
    test_g = tensor2numpy(test_gauss) # Bx7x7

    scores = tensor2numpy(scores) # Bx7x7

    prediction_mask = tensor2numpy(F.softmax(prediction_mask, dim=1))
    pred_fg = prediction_mask[:, :, 0]
    # pred_bg = prediction_mask[:, :, 1]

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    imshow(axes[0, 0], train_rgb, title='train rgb')
    add_rect(axes[0, 0], train_b, color='r')
    imshow(axes[0, 1], train_d, title='depth')
    imshow(axes[0, 2], train_m, title='mask')

    imshow(axes[1, 0], test_rgb, title='test rgb')
    add_rect(axes[1, 0], test_b, color='r')
    imshow(axes[1, 1], test_d, title='depth')
    imshow(axes[1, 2], test_m, title='mask')


    imshow(axes[2, 0], test_g, title='test score')
    imshow(axes[2, 1], scores, title='pred score : %f'%np.max(scores))
    imshow(axes[2, 2], pred_fg, title='pred mask')

    plt.savefig(os.path.join(temp_path,'debug-%d-%d.png'%(epoch, batch)))
    plt.close(fig)
