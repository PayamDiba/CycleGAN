"""
@author: Payam Dibaeinia, and {add your names}
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from ResNetBlock import ResNetBlock


class generator(nn.Module):
    """
    According to the cycleGAN paper, reflection padding and instance normalization was used.
    Also all dimensions including number of channels and kernels were set to those defined in cycleGAN paper

    Arbitrary selections of hyper_parameters:
    - Use bias in both convolutional layers
    - If use dropout, prob = 0.5
    """

    def __init__(self,in_channels, out_channels, nBlocks, nChanFirstConv = 64, dropout = False):
        """
        nChanFirstConv: number of channels of the first convolution layer, 64 used in cycleGAN paper
        """
        super(generator,self).__init__()

        layers = []
        layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(in_channels = in_channels, out_channels = nChanFirstConv, kernel_size = 7)]
        layers += [nn.InstanceNorm2d(nChanFirstConv)]
        layers += [nn.ReLU(inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv, out_channels = nChanFirstConv*2, kernel_size = 3, stride = 2, padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*2)]
        layers += [nn.ReLU(inplace = True)]

        layers += [nn.Conv2d(in_channels = nChanFirstConv*2, out_channels = nChanFirstConv*4, kernel_size = 3, stride=2, padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*4)]
        layers += [nn.ReLU(inplace = True)]

        for i in range(nBlocks):
            layers += [ResNetBlock(nChanFirstConv*4, dropout)]

        layers += [nn.ConvTranspose2d(in_channels = nChanFirstConv*4, out_channels = nChanFirstConv*2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv*2)]
        layers += [nn.ReLU(inplace = True)]

        layers += [nn.ConvTranspose2d(in_channels = nChanFirstConv*2, out_channels = nChanFirstConv, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)]
        layers += [nn.InstanceNorm2d(nChanFirstConv)]
        layers += [nn.ReLU(inplace = True)]

        layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(in_channels = nChanFirstConv, out_channels = out_channels, kernel_size = 7)]
        layers += [nn.Tanh()]

        self.all_layers_ = nn.Sequential(*layers)

    def forward(self,x):
        return self.all_layers_(x)
