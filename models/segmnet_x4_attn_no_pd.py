import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
from collections import OrderedDict

try:
    from .resnet import Bottleneck, resnet50
    from .query_decoder_x4_attn_no_pd import QueryDeCoder, Embeddings
except:
    from resnet import Bottleneck, resnet50
    from query_decoder_x4_attn_no_pd import QueryDeCoder, Embeddings

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

'''
Query decoder also output attention map, used as DCF for localization information
without pixel decoder
Input -> ResNet50 -> feature maps -> query decoder -> target query
target query + pixel feature -> mask
'''
class SegmNet(nn.Module):
    def __init__(self, backbone, query_decoder,
                       backbone_layers=['layer1', 'layer2', 'layer3', 'layer4']):
        super(SegmNet, self).__init__()

        self.backbone = backbone
        self.query_decoder = query_decoder
        self.backbone_layers = backbone_layers

        # Convert features into BxPatchesxC, Bx49x2048, Bx196x1024, Bx196x512, Bx784x256
        self.feat_sz = 7
        self.feat_dim = 2048
        feat_sizes = [self.feat_sz * s for s in [1, 2, 4, 8]] # 7, 14, 28, 56
        feat_dims = [self.feat_dim // s for s in [1, 2, 4, 8]] # 2048, 1024, 512, 256
        patch_sizes = [1, 1, 2, 2]                             # result Patches: 7*7, 14*14, 14*14, 28*28
        self.token_sizes = [feat_sizes[i]//patch_sizes[i] for i in range(4)] # patch resolution from Embedding tokens

        self.embeddings = nn.ModuleList([])
        for f_s, p_s, f_d in zip(feat_sizes, patch_sizes, feat_dims):
            self.embeddings.append(Embeddings(img_size=(f_s, f_s), patch_size=(p_s, p_s), hidden_size=f_d, in_channels=f_d))

        ''' now fg_roi output 1, 16, 64, 144 queries, how about just output 1, 1, 1, 1 queries for each feature map ??? '''
        fg_roi_dims = [1, 4, 8, 12]
        fg_roi_scales = [1/32, 1/16, 1/8, 1/4]
        self.fg_query_layers = nn.ModuleList([])
        for roi_dim, roi_scale in zip(fg_roi_dims, fg_roi_scales):
            self.fg_query_layers.append(RoIPool((roi_dim, roi_dim), roi_scale))

        bg_roi_dims = [1, 3, 5, 7]
        self.bg_query_layers = nn.ModuleList([])
        for roi_dim in bg_roi_dims:
            self.bg_query_layers.append(nn.AdaptiveAvgPool2d(roi_dim))

        # seperatable convolution, 1) convolution on each input channel, then use 1x1 conv
        # # Bx256x56x56 -> Bx256x56x56
        self.sep_conv = nn.Sequential(nn.Conv2d(self.feat_dim//8, self.feat_dim//8, kernel_size=5, padding=2, groups=self.feat_dim//8),
                                      nn.Conv2d(self.feat_dim//8, self.feat_dim//8, kernel_size=1))

        '''We use a cross-attn as prediction head '''
        self.hidden_dim = 256
        self.num_heads = 8
        self.fg_q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bg_q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.kv_linear = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.normalize_fact = float(self.hidden_dim / self.num_heads) ** -0.5

        #
        self.post2 = conv(512, 64)
        self.post1 = conv(64, 16)
        self.post0 = conv_no_relu(16, 2)

    def extract_backbone_feat(self, color, depth):
        color_feat = self.backbone(color)
        depth_feat = self.backbone(depth)
        rgbd_feat = [torch.max(color_feat[key], depth_feat[key]) for key in self.backbone_layers]
        return rgbd_feat[::-1]

    def flatten_feat(self, f):
        ''' return [B, Pixels, C]'''
        return f.view(f.shape[0], f.shape[1], -1).transpose(-1, -2)

    def masked_feat(self, feat, mask):
        batch, channels, height, width = feat.shape
        fg_feat = feat * F.interpolate(mask, size=(height, width))     # B, C, 7, 7
        bg_feat = feat * F.interpolate(1-mask, size=(height, width))
        return fg_feat, bg_feat

    def init_queries(self, feat_rgbd, mask, rois):
        '''Input : resnet50 feature maps : [Bx256x56x56, Bx512x28x28, Bx1024x14x14, Bx2048x7x7]
          Output : fg_query = [Bx144x256, Bx64x512, Bx16x1024, Bx1x2048]
                   bg_query = [Bx49x256,  Bx25x512, Bx9x1024,  Bx1x2048]
        '''
        fg_queries, bg_queries = [], []
        for layer, f_rgbd in enumerate(feat_rgbd):
            f_fg, f_bg = self.masked_feat(f_rgbd, mask)
            fg_queries.append(self.flatten_feat(self.fg_query_layers[layer](f_fg, rois)))
            bg_queries.append(self.flatten_feat(self.bg_query_layers[layer](f_bg)))
        return fg_queries, bg_queries

    def mask_head(self, test_feat, fg_query, bg_query, score):
        ''' mask = softmax(test_feat * query)
            mask = conv_layers(mask)
            similar as TransTrack, MHAttentionMap, a 2D attention module
        '''
        score = F.interpolate(score, size=(test_feat.shape[-1], test_feat.shape[-2]))
        test_feat = test_feat + score # Bx256x56x56

        # 2D attention module
        kv = self.kv_linear(test_feat)    # Bx256x56x56
        fg_q = self.fg_q_linear(fg_query)
        bg_q = self.bg_q_linear(bg_query)
        #
        fg_qh = fg_q.view(fg_q.shape[0], fg_q.shape[1], self.num_heads, fg_q.shape[2]//self.num_heads)
        bg_qh = bg_q.view(bg_q.shape[0], bg_q.shape[1], self.num_heads, bg_q.shape[2]//self.num_heads)
        kv = kv.view(kv.shape[0], self.num_heads, kv.shape[1]//self.num_heads, kv.shape[-2], kv.shape[-1])
        # cross attention
        fg_mask = torch.einsum("bqnc,bnchw->bqnhw", fg_qh * self.normalize_fact, kv)
        fg_mask = F.softmax(fg_mask.flatten(2), dim=-1).view_as(fg_mask) # BxQxNxHxW -> BxQxNHW -> softmax -> BxQxNxHxW
        fg_mask = self.dropout(fg_mask) # BxQxNxHxW
        #
        bg_mask = torch.einsum("bqnc,bnchw->bqnhw", bg_qh * self.normalize_fact, kv)
        bg_mask = F.softmax(bg_mask.flatten(2), dim=-1).view_as(bg_mask) # BxQxNxHxW -> BxQxNHW -> softmax -> BxQxNxHxW
        bg_mask = self.dropout(bg_mask) # BxQxNxHxW

        # Post processing
        fg_mask = torch.sum(torch.mean(fg_mask, dim=2), dim=1) # BxHxW
        bg_mask = torch.sum(torch.mean(bg_mask, dim=2), dim=1)

        mask = torch.cat((fg_mask.unsqueeze(1)+test_feat, bg_mask.unsqueeze(1)+test_feat), dim=1) # Bx2*256xHxW
        mask = F.interpolate(self.post2(mask), scale_factor=2) # Bx512 -> 64
        mask = self.post1(mask) # Bx64x112x112 -> Bx16x112x112
        mask = self.post0(mask) # Bx2x112x112
        return mask


    def forward(self, train_color, train_depth, train_mask, train_rois, test_color, test_depth):
        ''' resnet50 : [Bx256x56x56, Bx512x28x28, Bx1024x14x14, Bx2048x7x7] -> reversed
        '''
        # Learn the target query
        train_rgbd_feat = self.extract_backbone_feat(train_color, train_depth)  # [layer4, layer3, layer2, layer1]
        test_rgbd_feat = self.extract_backbone_feat(test_color, test_depth)

        # initial FG and BG queries for multi layers
        fg_queries, bg_queries = self.init_queries(train_rgbd_feat, train_mask, train_rois)

        # Update the queires
        test_pixel_feat = [self.embeddings[i](test_rgbd_feat[i]) for i in range(4)] # convert to Patches: 7*7, 14*14, 14*14, 28*28
        fg_queries, fg_attns = self.query_decoder(fg_queries, test_pixel_feat) # Q: [B, Q=1+16+64+144, C=256], attn : [B, H*Q=8*144, P=784]
        bg_queries, _ = self.query_decoder(bg_queries, test_pixel_feat)        # Q: [B, Q=1+9+25+49=84, C=256]

        # fg_attns : [B, H, Q, P=784], if we only use the last attention map
        score = fg_attns.view(fg_attns.shape[0], self.token_sizes[-1], self.token_sizes[-1]) # [B, 28, 28]

        # Mask prediction head
        test_kv = self.sep_conv(test_rgbd_feat[-1]) # Bx256x56x56

        mask = self.mask_head(test_kv, fg_queries, bg_queries, score.unsqueeze(1))

        return mask, score


def create_segmnet(pretrained_backbone=False,
                   backbone_output_layers=['layer1', 'layer2', 'layer3', 'layer4']):

    backbone = resnet50(output_layers=backbone_output_layers, pretrained=pretrained_backbone)
    query_decoder = QueryDeCoder()

    return SegmNet(backbone, query_decoder, backbone_layers=backbone_output_layers)


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
