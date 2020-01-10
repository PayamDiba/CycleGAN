"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ResNetBlock(nn.Module):
    """
    According to the cycleGAN paper, reflection padding and instance normalization was used

    Arbitrary selections of hyper_parameters:
    - Use bias in both convolutional layers
    - If use dropout, prob = 0.5
    """

    def __init__(self,nChannels, dropout = False):
        super(ResNetBlock,self).__init__()

        layers = []
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(in_channels = nChannels, out_channels = nChannels, kernel_size = 3)]
        layers += [nn.InstanceNorm2d(nChannels)]
        layers += [nn.ReLU(inplace = True)]

        if dropout:
            layers += [nn.Dropout(0.5)]

        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(in_channels = nChannels, out_channels = nChannels, kernel_size = 3)]
        layers += [nn.InstanceNorm2d(nChannels)]

        self.block_ = nn.Sequential(*layers)

    def forward(self,x):
        """
        residual connection is added
        """
        ret = x + self.block_(x)
        return ret
