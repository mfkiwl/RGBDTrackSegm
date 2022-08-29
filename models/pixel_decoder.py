"""Pixel Decoder for kMaX.
  The pixel decoder in the k-means Mask Transformer (kMaX) employs a simple
  decoder structure, similar to MaX-DeepLab-S model. We support using
  axial-block and bottleneck-block in the decoder, along with skip connections
  from the pixel encoder (i.e., backbone). When self-attention operations are
  used (e.g., axial-blocks), it is equivalent to incorporating the transformer
  encoder to the pixel decoder.
  References:
    [1] k-means Mask Transformer, ECCV 2022.
          Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
          Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

Pytorch version by Song Yan
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

try:
    from .resnet import Bottleneck, resnet50
except:
    from resnet import Bottleneck, resnet50

class PixelDecoder(nn.Module):
    def __init__(self, block, layers, output_layers, inplanes=2048, dilation_factor=1,
                       attention=[True, True, False, False], num_heads=8, width_per_group=64, image_size=224, inference=False):
        super(PixelDecoder, self).__init__()
        self.inplanes = inplanes
        self.output_layers = output_layers
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, inplanes//4, layers[0], stride=1, dilation=dilation_factor,
                                       attention=attention[0], num_heads=num_heads, kernel_size=image_size//32) # 2048, 1/32
        self.layer2 = self._make_layer(block, inplanes//8, layers[1], stride=1, dilation=dilation_factor,
                                       attention=attention[1], num_heads=num_heads, kernel_size=image_size//16) # 1024, 1/16
        self.layer3 = self._make_layer(block, inplanes//16, layers[2], stride=1, dilation=dilation_factor,
                                       attention=attention[2], num_heads=num_heads, kernel_size=image_size//8) # 512, 1/8
        self.layer4 = self._make_layer(block, inplanes//32, layers[3], dilation=dilation_factor,
                                       attention=attention[3], num_heads=num_heads, kernel_size=image_size//4) # 256, 1/4


        # seperatable convolution, 1) convolution on each input channel, then use 1x1 conv
        self.sep_conv = nn.Sequential(nn.Conv2d(inplanes//8, inplanes//8, kernel_size=5, padding=2, groups=inplanes//8),
                                      nn.Conv2d(inplanes//8, inplanes//8, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, attention=False, num_heads=8, kernel_size=56, inference=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation,
                            attention=attention, num_heads=num_heads, kernel_size=kernel_size,
                            inference=inference, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                attention=attention, num_heads=num_heads, kernel_size=kernel_size,
                                inference=inference, base_width=self.base_width))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, features, output_layers=None):
        '''
        features, ResNet50 with axial blocks features, [C2, C3, C4, C5]
        output_layers, [layer1, layer2, layer3, sep]
        '''

        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.layer1(features[3]) # C5, Bx2048x7x7

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = F.interpolate(x, scale_factor=2) # Bx2048x14x14
        x = self.layer2(x) + features[2]  # C4, Bx1024x14x14

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = F.interpolate(x, scale_factor=2) # Bx1024x28x28
        x = self.layer3(x) + features[1]  # C3, Bx512x28x28

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = F.interpolate(x, scale_factor=2) # Bx512x56x56
        x = self.layer4(x) + features[0]  # C2, Bx256x56x56

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = F.interpolate(x, scale_factor=2) # Bx256x112x112
        x = self.sep_conv(x)                 # Bx256x112x112

        if self._add_output_and_check('sep', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def create_pixel_decoder(layers=[1, 5, 1, 1], attention=[True, True, False, False],
                         output_layers=['layer1', 'layer2', 'layer3', 'sep'], **kwargs):
    """ Constructs a PixelDecoder in KMax_DeepLab2"""
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['layer1', 'layer2', 'layer3', 'layer4', 'sep']:
                raise ValueError('PixelDecorder Unknown layer: {}'.format(l))

    model = PixelDecoder(Bottleneck, layers, output_layers, **kwargs)
    return model

if __name__ == '__main__':

    device = torch.device('cuda')

    model = create_pixel_decoder()
    model = model.to(device)

    input = [torch.zeros(8, 256, 56, 56).to(device),  # ResNet50 Layer1
             torch.zeros(8, 512, 28, 28).to(device),  # ResNet50 Layer2
             torch.zeros(8, 1024, 14, 14).to(device), # ResNet50 Layer3
             torch.zeros(8, 2048, 7, 7).to(device)]   # ResNet50 Layer4

    output = model(input)

    # # layer1 torch.Size([Batch, 2048, 7, 7])
    # # layer2 torch.Size([Batch, 1024, 14, 14])
    # # layer3 torch.Size([Batch, 512, 28, 28])
    # # sep torch.Size([Batch, 256, 112, 112])
    #
    for feat in output:
        print(feat, output[feat].shape)
