import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

try:
    from .resnet import Bottleneck, resnet50
    from .pixel_decoder import create_pixel_decoder
    from .query_decoder_x4 import QueryDeCoder
except:
    from resnet import Bottleneck, resnet50
    from pixel_decoder import create_pixel_decoder
    from query_decoder_x4 import QueryDeCoder

from torchvision.ops import RoIPool

def conv1x1(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                      padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))

class SegmNet(nn.Module):
    def __init__(self, backbone, pixel_decoder, query_decoder,
                       backbone_layers=['layer1', 'layer2', 'layer3', 'layer4'],
                       pixel_decoder_layers=['layer1', 'layer2', 'layer3', 'layer4', 'sep']):
        super(SegmNet, self).__init__()

        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.query_decoder = query_decoder
        self.backbone_layers = backbone_layers
        self.pixel_decoder_layers = pixel_decoder_layers

        self.feat_pool = nn.AdaptiveAvgPool2d(7) # output size is 7

        fg_roi_dims = [1, 4, 8, 12]
        fg_roi_scales = [1/32, 1/16, 1/8, 1/4]
        self.fg_query_layers = nn.ModuleList([])
        for roi_dim, roi_scale in zip(fg_roi_dims, fg_roi_scales):
            self.fg_query_layers.append(RoIPool((roi_dim, roi_dim), roi_scale))

        bg_roi_dims = [1, 3, 5, 7]
        self.bg_query_layers = nn.ModuleList([])
        for roi_dim in bg_roi_dims:
            self.bg_query_layers.append(nn.AdaptiveAvgPool2d(roi_dim))

        # For RGBD feature fusion, how about memory????
        spatial_dim = 256
        self.spatial_layers = nn.ModuleList([])
        for _ in range(4):
            self.spatial_layers.append(nn.ModuleList([conv1x1(spatial_dim, spatial_dim),
                                                      conv1x1(spatial_dim, spatial_dim),
                                                      conv1x1(spatial_dim, spatial_dim)]))
            spatial_dim = spatial_dim * 2

        self.post_layers = nn.ModuleList([conv(309, 32), conv(32, 16), conv_no_relu(16, 2)])

    def spatial_attention(self, f_d1, f_d2, f_rgb):
        f_d1 = f_d1.view(f_d1.shape[0], f_d1.shape[1], -1) # BxCxP
        f_d2 = f_d2.view(f_d2.shape[0], f_d2.shape[1], -1) # BxCxP
        w_d = torch.matmul(f_d1.transpose(-1, -2), f_d2) # BxPxP, Bx3136x3136, 56^2, 28^2, 14^2, 7^2
        w_d = F.softmax(w_d, dim=1) # ????

        B, C, H, W = f_rgb.shape
        f_rgb = f_rgb.view(f_rgb.shape[0], f_rgb.shape[1], -1) # BxCxP
        f_rgbd = torch.matmul(f_rgb, w_d) + f_rgb
        f_rgbd = f_rgbd.view(B, C, H, W)
        return f_rgbd

    def extract_backbone_feat(self, color, depth):
        feat_rgb = self.backbone(color)
        feat_rgb = [f for f in feat_rgb.values()] # 256x56x56, 512x28x28, 1024x14x14, 2048x7x7
        feat_d = self.backbone(depth)
        feat_d = [f for f in feat_d.values()]

        ''' use spatial attention for RGBD features fusion '''
        feat_rgbd = []
        for f_rgb, f_d, (conv_d1, conv_d2, conv_rgb) in zip(feat_rgb, feat_d, self.spatial_layers):
            feat_rgbd.append(self.spatial_attention(conv_d1(f_d), conv_d2(f_d), conv_rgb(f_rgb)))

        return feat_rgbd

    def flatten_pixel_features(self, f):
        ''' Input : [B, C, 7, 7], [B, C, 14, 14], [B, C, 28, 28], [B, C, 56, 56]
            Return : use adaptiveAvgPool2d, -> [B, C, 7, 7] -> [B, 49, C]
        '''
        return [self.flatten_feat(self.feat_pool(f[key])) for key in self.pixel_decoder_layers if key != 'sep']

    def flatten_feat(self, f):
        ''' return [B, Pixels, C]'''
        return f.view(f.shape[0], f.shape[1], -1).transpose(-1, -2)

    def masked_feat(self, feat, mask):
        batch, channels, height, width = feat.shape
        fg_feat = feat * F.interpolate(mask, size=(height, width))     # B, C, 7, 7
        bg_feat = feat * F.interpolate(1-mask, size=(height, width))
        return fg_feat, bg_feat

    def init_queries(self, feat_rgbd, mask, rois):
        ''' train_rgbd_feat, features from pixel_decoder
            layer1 layer2 layer3 (layer4) sep
            [Bx2048x7x7, Bx1024x14x14, Bx512x28x28, Bx256x56x56, Bx256x112x112]
        return : fg_query=[Bx1x2048, Bx16x1024, Bx64x512], bg_query=[Bx1x2048, Bx9x1024, Bx9x512]
        '''
        fg_queries, bg_queries = [], []
        for layer, f_rgbd in enumerate(feat_rgbd):
            f_fg, f_bg = self.masked_feat(f_rgbd, mask)
            q_fg = self.flatten_feat(self.fg_query_layers[layer](f_fg, rois))
            q_bg = self.flatten_feat(self.bg_query_layers[layer](f_bg))

            fg_queries.append(q_fg)
            bg_queries.append(q_bg)
        return fg_queries, bg_queries

    def prediction_head(self, test_feat, fg_query, bg_query):
        ''' simple FFN
        '''
        pixels = test_feat.view(test_feat.shape[0], test_feat.shape[1], -1) # [B, 256, 112*112]
        #
        fg_mask = torch.matmul(fg_query, pixels) # [B, Q=1+16+64+144=255, C] * [B, C, Pixels] -> [B, Q, Pixels]
        bg_mask = torch.matmul(bg_query, pixels) # [B, Q=1+9+9, C]

        mask = torch.cat((fg_mask, bg_mask), dim=1)
        mask = mask.view(mask.shape[0], mask.shape[1], test_feat.shape[2], test_feat.shape[3])

        for post in self.post_layers:
            mask = post(mask)

        return mask


    def forward(self, train_color, train_depth, train_mask, train_rois, test_color, test_depth):
        ''' pixel decoder : [layer1, layer2, layer3, sep]
            layer1 : Bx2048x7x7, layer2 : Bx1024x14x14, layer3 : Bx512x28x28, sep : Bx256x112x112
            # layer4 : Bx256x56x56 # no layer4
        '''
        # Encoded pixel features from test frame
        test_rgbd_feat = self.extract_backbone_feat(test_color, test_depth)
        test_rgbd_feat = self.pixel_decoder(test_rgbd_feat)
        # convert all features to [B, 49, C] for small memory
        test_pixel_feat = self.flatten_pixel_features(test_rgbd_feat)

        # Learn the target query
        train_rgbd_feat = self.extract_backbone_feat(train_color, train_depth)
        train_rgbd_feat = self.pixel_decoder(train_rgbd_feat)
        train_rgbd_feat = [f for f in train_rgbd_feat.values()] # layer1 layer2 layer3 layer4 sep

        # initial FG and BG queries for multi layers
        fg_queries, bg_queries = self.init_queries(train_rgbd_feat[:4], train_mask, train_rois)

        # Update the queires
        fg_queries = self.query_decoder(fg_queries, test_pixel_feat) # [B, 1+16+64+144=225, 256]
        bg_queries = self.query_decoder(bg_queries, test_pixel_feat) # [B, 1+9+25+49=84, 256]

        # Mask prediction head
        mask = self.prediction_head(test_rgbd_feat['sep'], fg_queries, bg_queries)

        return mask


def create_segmnet(pretrained_backbone=False,
                   backbone_output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
                   pixel_decoder_output_layers=['layer1', 'layer2', 'layer3', 'layer4', 'sep']):

    backbone = resnet50(output_layers=backbone_output_layers, pretrained=pretrained_backbone)
    pixel_decoder = create_pixel_decoder(layers=[1, 5, 1, 1], attention=[True, True, False, False],
                                         output_layers=pixel_decoder_output_layers)
    query_decoder = QueryDeCoder()

    return SegmNet(backbone, pixel_decoder, query_decoder,
                   backbone_layers=backbone_output_layers,
                   pixel_decoder_layers=pixel_decoder_output_layers)


if __name__ == '__main__':

    model = create_segmnet()
    model = model.cuda()

    batch_size = 16
    img_sz = 224
    train_color = torch.zeros(batch_size, 3, img_sz, img_sz).cuda()
    train_depth = torch.zeros(batch_size, 3, img_sz, img_sz).cuda()
    train_mask = torch.zeros(batch_size, 1, img_sz, img_sz).cuda()
    test_color = torch.zeros(batch_size, 3, img_sz, img_sz).cuda()
    test_depth = torch.zeros(batch_size, 3, img_sz, img_sz).cuda()

    output = model(train_color, train_depth, train_mask, test_color, test_depth)
    print(output.shape)
