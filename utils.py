"""
@author: Payam Dibaeinia
"""

import torch
from torch import autograd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np


def plot(samples):
    """
    samples: pytorch tensor containing the output of a generator
    As of now it cannot handle more than 25 images
    """
    #DEBUG: the below command might raise error, not sure if samples is going
    # to be a tensor or list of tensors or array! For now assume it is tensor
    nImages = np.shape(samples)[0]
    assert(nImages <= 25)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    return fig

def calculate_lr(initial_lr, steps_constant, steps_decay, total_steps, currEpoch):
    """
    calculates new learning rate:
    over the first 'steps_constant' steps it returns the same initial learning rate
    over the next 'steps_decay' steps it linearly decays the learning rate
    total_steps = steps_constant + steps_decay

    currEpoch: the training epoch that was just finished


    e.g. :
    over epochs 0 - 99 (includisve): LR = initial LR
    ovrt epochs 100 - 199 (inclusive): LR decays to zero
    Note that LR for epoch 100 is determined when currEpoch = 99
    Usage at epoch 10: calculate_lr(0.001, 100, 100, 200, 10)
    """
    assert(steps_constant + steps_decay == total_steps)

    a = max(0, currEpoch - (steps_constant - 2))
    b = float(steps_decay) + 1.0
    c = np.true_divide(a,b)

    return (1-c) * initial_lr

def make_dir(flags):
    """
    Creates the given directory if does not already exists
    """
    path_data = flags.data_dir
    path_model = flags.checkpoint_dir
    path_images = flags.image_dir

    if os.path.exists(path_data) and os.path.exists(path_model) and os.path.exists(path_images):
        raise RuntimeError('All of the given directories already exist. It will over-write the results. Please take backup and delete the existing models and images directories or move to a new directory and try again.')

    if not os.path.exists(path_data):
        os.mkdir(path_data)

    if not os.path.exists(path_model):
        os.mkdir(path_model)

    if not os.path.exists(path_images):
        os.mkdir(path_images)
        os.mkdir(path_images + '/train')
        os.mkdir(path_images + '/test')

def init_weights(model, type = 'normal', scale = 0.02):
    """
    Initializes the neural network weights.
    The function was taken from :
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def initilizer(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, scale)
            elif type == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight.data, gain = scale)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scale)
            torch.nn.init.constant_(m.bias.data, 0.0)

    model.apply(initilizer)
