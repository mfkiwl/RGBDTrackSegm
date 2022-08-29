import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                      padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4):
        self.spatial_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.spatial_layers.append(nn.ModuleList([conv1x1(hidden_dim, hidden_dim),
                                                      conv1x1(hidden_dim, hidden_dim),
                                                      conv1x1(hidden_dim, hidden_dim)]))
            hidden_dim = hidden_dim * 2

    def spatial_attention(self, f_d1, f_d2, f_rgb):
        ''' spatial attention over color and depth feature maps '''
        f_d1 = f_d1.view(f_d1.shape[0], f_d1.shape[1], -1) # BxCxP
        f_d2 = f_d2.view(f_d2.shape[0], f_d2.shape[1], -1) # BxCxP
        w_d = torch.matmul(f_d1.transpose(-1, -2), f_d2) # BxPxP, Bx3136x3136, 56^2, 28^2, 14^2, 7^2
        w_d = F.softmax(w_d, dim=1) ### should softmax along which axis???

        B, C, H, W = f_rgb.shape
        f_rgb = f_rgb.view(f_rgb.shape[0], f_rgb.shape[1], -1) # BxCxP
        f_rgbd = torch.matmul(f_rgb, w_d) + f_rgb
        f_rgbd = f_rgbd.view(B, C, H, W)
        return f_rgbd

    def forward(self, feat_rgb, feat_d):
        ''' feat_rgb, feat_d : ResNet50 feature maps '''
        feat_rgbd = []
        for f_rgb, f_d, (conv_d1, conv_d2, conv_rgb) in zip(feat_rgb, feat_d, self.spatial_layers):
            feat_rgbd.append(self.spatial_attention(conv_d1(f_d), conv_d2(f_d), conv_rgb(f_rgb)))
        return feat_rgbd
