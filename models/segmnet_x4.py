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

        self.fg_query_pool0 = RoIPool((1, 1), 1/32) # for layer1 in PixelDecoder, Bx2048x7x7, 1/32
        self.fg_query_pool1 = RoIPool((4, 4), 1/16) # for layer2 in PixelDecoder, Bx1024x14x14, 1/16
        self.fg_query_pool2 = RoIPool((8, 8), 1/8)  # for layer3 in PixelDecoder, Bx512x28x28, 1/8
        self.fg_query_pool3 = RoIPool((12, 12), 1/4) # for layer4 in PixelDecoder, Bx256x56x56, 1/4

        self.bg_query_pool0 = nn.AdaptiveAvgPool2d(1) #   7x7 -> 1x1
        self.bg_query_pool1 = nn.AdaptiveAvgPool2d(3) # 14x14 -> 3x3
        self.bg_query_pool2 = nn.AdaptiveAvgPool2d(5) # 28x28 -> 5x5
        self.bg_query_pool3 = nn.AdaptiveAvgPool2d(7) # 56x56 -> 7x7

        self.post2 = conv(309, 32) # 1+16+64+144 + 1+9+25+49 = 225 + 84 = 309
        self.post1 = conv(32, 16)
        self.post0 = conv_no_relu(16, 2)

    def extract_backbone_feat(self, color, depth):
        color_feat = self.backbone(color)
        depth_feat = self.backbone(depth)
        rgbd_feat = [torch.max(color_feat[key], depth_feat[key]) for key in self.backbone_layers]
        return rgbd_feat

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

    def init_queries(self, train_rgbd_feat, train_mask, train_rois):
        ''' train_rgbd_feat, features from pixel_decoder
            layer1 layer2 layer3 (layer4) sep
            [Bx2048x7x7, Bx1024x14x14, Bx512x28x28, Bx256x56x56, Bx256x112x112]
        return : fg_query=[Bx1x2048, Bx16x1024, Bx64x512], bg_query=[Bx1x2048, Bx9x1024, Bx9x512]
        '''
        # [B, 2048, 7, 7], 1/32 -> Bx2048x1x1 -> Bx1x2048
        fg_feat0, bg_feat0 = self.masked_feat(train_rgbd_feat[0], train_mask)
        fg_query0 = self.flatten_feat(self.fg_query_pool0(fg_feat0, train_rois))
        bg_query0 = self.flatten_feat(self.bg_query_pool0(bg_feat0))

        # [B, 1024, 14, 14], 1/16, -> Bx1024x4x4 -> Bx16x1024
        fg_feat1, bg_feat1 = self.masked_feat(train_rgbd_feat[1], train_mask)
        fg_query1 = self.flatten_feat(self.fg_query_pool1(fg_feat1, train_rois))
        bg_query1 = self.flatten_feat(self.bg_query_pool1(bg_feat1))

        # [B, 512, 28, 28], 1/8, -> Bx512x8x8 -> Bx64x512
        fg_feat2, bg_feat2 = self.masked_feat(train_rgbd_feat[2], train_mask)
        fg_query2 = self.flatten_feat(self.fg_query_pool2(fg_feat2, train_rois))
        bg_query2 = self.flatten_feat(self.bg_query_pool2(bg_feat2))

        # [B, 256, 56, 56], 1/4, -> Bx256x16x16 -> Bx144x256
        fg_feat3, bg_feat3 = self.masked_feat(train_rgbd_feat[3], train_mask)
        fg_query3 = self.flatten_feat(self.fg_query_pool3(fg_feat3, train_rois))
        bg_query3 = self.flatten_feat(self.bg_query_pool3(bg_feat3))

        return [fg_query0, fg_query1, fg_query2, fg_query3], [bg_query0, bg_query1, bg_query2, bg_query3]

    def prediction_head(self, test_feat, fg_query, bg_query):
        ''' simple FFN
        '''
        pixels = test_feat.view(test_feat.shape[0], test_feat.shape[1], -1) # [B, 256, 112*112]
        #
        fg_mask = torch.matmul(fg_query, pixels) # [B, Q=1+16+64+144=255, C] * [B, C, Pixels] -> [B, Q, Pixels]
        bg_mask = torch.matmul(bg_query, pixels) # [B, Q=1+9+9, C]

        mask = torch.cat((fg_mask, bg_mask), dim=1)
        mask = mask.view(mask.shape[0], mask.shape[1], test_feat.shape[2], test_feat.shape[3])

        mask = self.post0(self.post1(self.post2(mask)))
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
        train_rgbd_feat = [f for f in train_rgbd_feat.values()] #

        # initial FG and BG queries for multi layers
        fg_queries, bg_queries = self.init_queries(train_rgbd_feat, train_mask, train_rois)

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
