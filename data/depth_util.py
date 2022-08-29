
from fast_pytorch_kmeans import KMeans
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import torch.nn.functional as F
import math
import os
import random

def depth_projection(depth, n_clusters=3, mode='euclidean', verbose=0):
    ''' Depth : torch Tensor, HxW -> Kmeans, K*H*W
    projecting to different depth planes, k-clusters, default k=3
    fast_pytorch_kmeans from : https://github.com/DeMoriarty/fast_pytorch_kmeans
    '''
    try:
        H, W = depth.shape
    except:
        (H, W) = depth.shape

    x = depth.view(H*W, 1)

    kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=verbose)
    labels = kmeans.fit_predict(x)

    sorted_centroids, sorted_indices = torch.sort(kmeans.centroids, dim=0)

    output = torch.zeros((n_clusters, H, W))
    for i in range(n_clusters):
        sorted_c = sorted_centroids[i]
        sorted_i = sorted_indices[i]

        k_idx = (labels == sorted_i).unsqueeze(-1) # N*1

        k = x * k_idx
        k = k - sorted_c # mean
        k = (k-torch.min(k[k_idx])) / (torch.max(k[k_idx])-torch.min(k[k_idx]))
        # torch 1.12.0 torch.clip(); torch 1.4.0 torch.clamp()
        k = torch.clamp(k, min=0.0, max=1.0)

        output[i, :, :] = k.view(H, W)

    return output, sorted_centroids


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device : ', device)

    K = 3

    fig, axes = plt.subplots(2, K, figsize=(8, 8))

    dataset = 'Youtube-VOS-2018' # 'CDTB'
    sequences = os.listdir('/home/yan/Data4/Datasets/%s/train/JPEGImages/'%dataset)
    seq = random.choice(sequences) # seq = '0bcfc4177d' # 'XMG_outside'
    print(dataset, seq)
    data_path_format = '/home/yan/Data4/Datasets/%s/train/JPEGImages/%s/depth/'
    color_path_format = '/home/yan/Data4/Datasets/%s/train/JPEGImages/%s/'
    # data_path_format = '/home/yan/Data4/Datasets/%s/sequences/%s/depth/%08d.png'

    frames = os.listdir(data_path_format%(dataset, seq))

    for frame in frames:
        color_path = color_path_format%(dataset, seq) + frame[:-4]+'.jpg'
        color = cv2.imread(color_path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth_path = data_path_format%(dataset, seq)+frame
        depth = cv2.imread(depth_path, -1)
        H, W = depth.shape
        if H > W:
            pad = H - W
            depth = cv2.copyMakeBorder(depth, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)
            color = cv2.copyMakeBorder(color, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            pad = W - H
            depth = cv2.copyMakeBorder(depth, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
            color = cv2.copyMakeBorder(color, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

        depth = cv2.resize(depth, (224, 224), interpolation = cv2.INTER_AREA)
        color = cv2.resize(color, (224, 224), interpolation = cv2.INTER_AREA)

        depth = np.asarray(depth, dtype=np.float32)
        depth = torch.from_numpy(depth).to(device)

        output, centroids = depth_projection(depth, n_clusters=K, verbose=1)

        depth = depth.detach().clone().cpu().numpy()
        output = output.detach().clone().cpu().permute(1, 2, 0).numpy()

        plt.cla()
        axes[0, 0].imshow(depth)
        axes[0, 0].set_title('depth')

        if K == 1 or K == 3:
            axes[0, 1].imshow(output)
            axes[0, 1].set_title('new depth')

        axes[0, 2].imshow(color)
        axes[0, 2].set_title('Color')

        for k in range(K):
            axes[1, k].imshow(output[:, :, k])
            axes[1, k].set_title('Center : %f'%centroids[k])

        plt.pause(1)
        plt.show(block=False)
