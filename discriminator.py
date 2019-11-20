"""
@author: Payam Dibaeinia, and {add your names}
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class discriminator(nn.Module):
    """PatchGAN discriminator defined in cycleGAN paper

    According to the cycleGAN paper, instance normalization was used for all but first conv layer.
    Also all dimensions including number of channels and kernels were set to those defined in cycleGAN paper

    Arbitrary selections of hyper_parameters:
    - Use bias in both convolutional layers
    """

    def __init__(self,in_channels, nChanFirstConv = 64):
        """
        nChanFirstConv: number of channels of the first convolution layer, 64 used in cycleGAN paper
        """
        super(discriminator,self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels = in_channels, out_channels = nChanFirstConv, kernel_size = 4, stride = 2, padding = 1)]
        layers += [nn.LeakyReLU(negative_slope = 0.2, inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv, out_channels = nChanFirstConv*2, kernel_size = 4, stride = 2, padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*2)]
        layers += [nn.LeakyReLU(negative_slope = 0.2, inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv*2, out_channels = nChanFirstConv*4, kernel_size = 4, stride = 2, padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*4)]
        layers += [nn.LeakyReLU(negative_slope = 0.2, inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv*4, out_channels = nChanFirstConv*8, kernel_size = 4, stride = 1, padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*8)]
        layers += [nn.LeakyReLU(negative_slope = 0.2, inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv*8, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)]

        self.all_layers_ = nn.Sequential(*layers)

    def forward(self,x):
        return self.all_layers_(x)
