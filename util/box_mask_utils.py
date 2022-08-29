import numpy as np
import torch
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def mask2box(mask):
    ''' mask to axis-aligned bounding box (x0, y0, w, h)
    mask : H*W only one instance
    box : (4,) only one instance
    '''
    x1, x2, y1, y2 = 0, 0, 0, 0
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
    box = np.array([x1, y1, x2-x1, y2-y1])
    return box.astype(np.int32)

def box2mask(box, height, width):
    ''' box : np.array, (4, ), (x, y, w, h) -> mask : np.array, (H, W)'''
    mask = np.zeros((height, width), dtype=np.float32)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    return mask

def box2roi(box):
    ''' box : torch tensor, 1*4, (x, y, w, h) -> roi : torch tensor, 1*5, (batch_idx, x, y, w, h) '''
    roi = torch.zeros((box.shape[0], 5,))
    for i in range(box.shape[0]):
        roi[i, 0] = i
        roi[i, 1] = box[i, 0] # x0
        roi[i, 2] = box[i, 1] # y0
        roi[i, 3] = box[i, 0] + box[i, 2] # x1
        roi[i, 4] = box[i, 1] + box[i, 3] # y1
    return roi.to(box.device)

def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss

def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)

def box2gauss(box, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True, density=False, uni_bias=0):
    ''' Copied from pytracking, https://github.com/visionml/pytracking/ltr/data/processing_utils.py
    Generates the gaussian label function centered at target_bb, same as DiMP and ATOM
    box : torch tensor, B*4, (x, y, w, h) -> guass : torch tensor, BxHxW
    '''
    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz) # output_sz
    feat_sz = torch.Tensor(feat_sz)

    # box in image_sz
    target_center = box[:, 0:2] + 0.5 * box[:, 2:4]
    target_center_norm = (target_center - image_sz / 2) / image_sz

    # center in feat_sz
    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias

    return label

def box_to_frame(bbox, prev_bbox, scale_factor, search_area_factor):
    '''
    bbox : box coordinates in crop image
    prev_bbox : the crop bbox in image coordinates
    scale_factor : scale_factor
    '''
    bbox_in_crop = bbox / scale_factor
    # Crop image
    prev_x, prev_y, prev_w, prev_h = prev_bbox
    crop_sz = math.ceil(math.sqrt(prev_w*prev_h) * search_area_factor)
    crop_x0 = round(prev_x + 0.5*prev_w - crop_sz*0.5)
    crop_y0 = round(prev_y + 0.5*prev_h - crop_sz*0.5)

    bbox_in_frame = bbox_in_crop
    bbox_in_frame[0] = bbox_in_crop[0]+crop_x0
    bbox_in_frame[1] = bbox_in_crop[1]+crop_y0
    bbox_in_frame = [int(bb) for bb in bbox_in_frame]

    return bbox_in_frame

def max_contour_in_mask(mask):
    ''' select the largest contour '''
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]

    if len(cnt_area) > 0 and len(contours) != 0 and np.max(cnt_area) > 50:
        contour = contours[np.argmax(cnt_area)]  # use max area polygon
        polygon = contour.reshape(-1, 2) # if we want to use polygon

        mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)

    return mask


if __name__ == '__main__':

    search_area_factor = 4.0
    output_sigma_factor = 1/2
    sigma_factor = output_sigma_factor / search_area_factor # 1/16
    feature_sz = 7
    kernel_sz = 2
    image_sz = feature_sz * 32 # 16 # 18*16 = 288, ResNet50: 288 -> 144 -> 72 -> 36 -> 18, 16 times

    target_bb = torch.zeros((1, 4)) # bbox in image or in feat ?
    bbox = [50, 100, 30, 30] # (x, y, w, hq)
    target_bb[0, :] = torch.tensor(bbox) # in [224, 224]

    gauss_label = box2gauss(target_bb, sigma_factor, kernel_sz, feature_sz, image_sz,
                            end_pad_if_even=True, density=False, uni_bias=0)

    print(gauss_label.shape) # should be feature_sz * feature_sz

    gauss_label = gauss_label.clone().detach().cpu().numpy()[0, :, :]

    image = np.zeros((image_sz, image_sz))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                              linewidth=1, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    axes[1].imshow(gauss_label)
    plt.show()
