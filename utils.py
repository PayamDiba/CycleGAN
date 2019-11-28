"""
@author: Payam Dibaeinia, and {add your names}
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
    nImages = samples.size()[0]
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
    assert(steps_constant + steps_decay = total_steps)

    a = max(0, currEpoch - (steps_constant - 2))
    b = float(steps_decay) + 1.0
    c = np.true_divide(a,b)

    return (1-c) * initial_lr
