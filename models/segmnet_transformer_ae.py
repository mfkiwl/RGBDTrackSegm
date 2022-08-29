import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
from collections import OrderedDict

try:
    from .resnet import Bottleneck, resnet50
    from .spatial_attention import SpatialAttention
    from .convolution_layers import conv, conv1x1, conv_no_relu
    from .transformers import Encoder, Decoder, Embeddings
except:
    from resnet import Bottleneck, resnet50
    from spatial_attention import SpatialAttention
    from convolution_layers import conv, conv1x1, conv_no_relu
    from transformers import Encoder, Decoder, Embeddings


class SegmNet(nn.Module):
    def __init__(self, backbone, encoder, decoder, rgbd_fusion,
                       backbone_layers=['layer1', 'layer2', 'layer3', 'layer4']):
        super(SegmNet, self).__init__()

        self.backbone = backbone
        self.backbone_layers = backbone_layers

        self.encoder = encoder
        self.decoder = decoder
        self.rgbd_fusion = rgbd_fusion

        # extract template features from feature maps
        self.roi_layers = nn.ModuleList([RoIPool((12, 12), 1/4), RoIPool((8, 8), 1/8), RoIPool((4, 4), 1/16), RoIPool((1, 1), 1/32)])
        # convert search feature maps into patches
        self.embeddings = nn.ModuleList([Embeddings(img_size=(56, 56), patch_size=(7, 7), hidden_size=256, in_channels=256),
                                         Embeddings(img_size=(28, 28), patch_size=(7, 7), hidden_size=256, in_channels=512),
                                         Embeddings(img_size=(14, 14), patch_size=(7, 7), hidden_size=256, in_channels=1024),
                                         Embeddings(img_size=(7, 7), patch_size=(1, 1), hidden_size=1024, in_channels=2048)])


    def mask_head(self, feat):

        return mask


    def get_roi_feat(self, feat, rois):
        ''' Using PreciseRoIPooling for template features '''
        roi_feat = []
        for f, layer_ in zip(feat, self.roi_layers):
            roi_feat.append(layer_(f, rois))
        return roi_feat

    def get_embeddings(self, feat):
        ''' Converting Seach features to patches '''
        embeddings = []
        for f, layer_ in zip(feat, self.embeddings):
            embeddings.append(layer_(f))
        return embeddings

    def extract_backbone_feat(self, color, depth):
        ''' 1) extract ResNet50 features : 256x56x56, 512x28x28, 1024x14x14, 2048x7x7
            2) use spatial attention for RGBD features fusion '''
        feat_rgb = self.backbone(color)
        feat_d = self.backbone(depth)
        feat_rgb = [f for f in feat_rgb.values()]
        feat_d = [f for f in feat_d.values()]
        feat_rgbd = self.rgbd_fusion(feat_rgb, feat_d)
        return feat_rgbd

    def forward(self, train_color, train_depth, train_mask, train_rois, test_color, test_depth):
        # Template
        train_feat = self.extract_backbone_feat(train_color, train_depth)
        train_feat = self.get_roi_feat(train_feat, train_rois) # BxCxRxR
        template = [f.view(f.shape[0], f.shape[1], -1).transpose(-1, -2) for f in train_feat] # BxPxC

        # Search
        test_feat = self.extract_backbone_feat(test_color, test_depth)
        search = self.get_embeddings(test_feat) # BxPxC

        kv = [torch.cat((s, t), axis=1) for s, t in zip(template, search)] # Bx(S+T)xC

        ''' Which features we should use for transformer ???? '''
        encoded_feat = self.encoder(kv)           # BxPatchesxC -> BxPatchesxC

        query = nn.Parameter(torch.zeros(test_feat.shape[0], 1, 1024))

        query = self.decoder(query, encoded_feat)  # BxPatchesxC -> BxPatchesxC

        mask, score = self.mask_head(query, encoded_feat)

        return mask, score


def create_segmnet(pretrained_backbone=False,
                   backbone_output_layers=['layer1', 'layer2', 'layer3', 'layer4']):

    backbone = resnet50(output_layers=backbone_output_layers, pretrained=pretrained_backbone)
    rgbd_fusion = SpatialAttention()
    encoder = Encoder(num_layers=6, hidden_size=1024, mlp_size=512, dropout=0.1)
    decoder = Decoder(num_layers=6, hidden_size=1024, mlp_size=512, dropout=0.1)

    return SegmNet(backbone, encoder, decoder, rgbd_fusion, backbone_layers=backbone_output_layers)


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

    output, scores = model(train_color, train_depth, train_mask, test_color, test_depth)
    print(output.shape)
