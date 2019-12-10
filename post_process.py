"""
@author: Payam Dibaeinia, and {add your names}
"""

import os
import torchvision.datasets.utils as utils
from zipfile import ZipFile
import shutil
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import torch
import numpy as np
from cycleGAN import cycleGAN
from absl import flags
from absl import logging
from absl import app
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def getDataLoader_postProcess(path, batch_size, transform = None, nWorkers = 1):
    if not transform:
        transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    datasetA = ImageFolderWithPaths(path + '/A/', transform = transform)
    datasetB = ImageFolderWithPaths(path + '/B/', transform = transform)

    trainLoaderA = torch.utils.data.DataLoader(datasetA, batch_size = batch_size, shuffle = False, num_workers = nWorkers)
    trainLoaderB = torch.utils.data.DataLoader(datasetB, batch_size = batch_size, shuffle = False, num_workers = nWorkers)

    return trainLoaderA, trainLoaderB



def plot_pp(sample):
    nImages = np.shape(sample)[0]
    assert(nImages == 1)

    fig = plt.figure()
    #gs = gridspec.GridSpec(1, 1)
    #gs.update(wspace=0, hspace=0)

    for i, s in enumerate(sample):
        ax = plt.gca()
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(s)

    return fig


FLAGS = flags.FLAGS

flags.DEFINE_string('glt', 'lse', 'GAN loss type; the type of loss function used in training: lse (Least Square Error, default), bce (Binary Cross Entropy), was (Wasserstein GAN)')
flags.DEFINE_float('l_idnt', 0.5, 'The relative weight of identity loss to cycle loss in the objective. Set to zero if there is no need to use identity loss')
flags.DEFINE_float('l_A', 10.0, 'The relative weight of forward cycle loss to GAN loss in the objective')
flags.DEFINE_float('l_B', 10.0, 'The relative weight of backward cycle loss to GAN loss in the objective')
flags.DEFINE_integer('ncA', 3, 'Number of channels in the A domain')
flags.DEFINE_integer('ncB', 3, 'Number of channels in the B domain')
flags.DEFINE_integer('nbl', 9, 'Number of residual blocks in the generators')
flags.DEFINE_integer('ncFirstConv', 64, 'Number of channels in the first convolution layer of generators')
flags.DEFINE_boolean('dropout', False, 'Whether use dropout in the residual blocks')
flags.DEFINE_float('lr', 0.0002, 'Initial learning rate')
flags.DEFINE_string('ds', None, 'Name of the dataset: horse2zebra, apple2orange, summer2winter_yosemite')
flags.DEFINE_string('data_dir', 'data', 'Directory of data')
flags.DEFINE_string('checkpoint_dir', 'models', 'Directory of saved checkpoints during training')
flags.DEFINE_string('image_dir', 'generated_images', 'Directory of generated images during training')
flags.DEFINE_integer('bs', 1, 'Batch size')
flags.DEFINE_integer('nw', 2, 'Number of workers when building dataLoaders')
flags.DEFINE_integer('nSamples', 10, 'Number of samples from train and test data to evaluate during training')
flags.DEFINE_integer('nEpoch', 200, 'Number of total training epochs')
flags.DEFINE_boolean('resume', False, 'Whether we are resuming training from a checkpoint')
flags.DEFINE_integer('last_epoch', None, 'Need to be specified if resuming from a checkpoint to determine the epoch from which training is continued. It is used to read the saved checkpoint')
flags.DEFINE_integer('freq', 5, 'Epoch frequency for saving model and evaluation on sampled images')
flags.DEFINE_integer('steps_constLR', 100, 'Number of the intial training steps (epochs) over which learning rate is constant')
flags.DEFINE_string('init_type', 'normal', 'The type of free parameters initilizer: normal, xavier_normal')
flags.DEFINE_float('init_scale', 0.02, 'Scale/gain that is used in free parameters initilizer')
flags.DEFINE_string('pp_dir', None, 'Directory of data for post processing')
flags.DEFINE_integer('pp_epoch', None, 'Epoch of the model to be loaded for post processing')
flags.DEFINE_string('pp_write', None, 'Directory of data for post processing')


def main(argv):

    """
    Data
    """
    A, B = getDataLoader_postProcess(path = FLAGS.pp_dir, batch_size = 1, transform = None, nWorkers = 1)

    """
    Load model
    """
    path_load = FLAGS.checkpoint_dir + '/checkpoint_' + str(FLAGS.pp_epoch) + '.tar'
    model = cycleGAN(FLAGS)
    model.load(path_load)


    """
    generate fake images
    """
    for a, b in zip(A, B):
        realA = a[0].cuda()
        realB = b[0].cuda()

        pathA = a[2][0]
        pathB = b[2][0]

        nameA = pathA.split('/')[-1]
        nameB = pathB.split('/')[-1]

        nameA = nameA.split('.')[0]
        nameB = nameB.split('.')[0]


        with torch.no_grad():
            model.gA.eval()
            model.gB.eval()
            fakesB = model.gA(realA)
            fakesA = model.gB(realB)

            fakesB = fakesB.data.cpu().numpy()
            fakesA = fakesA.data.cpu().numpy()
            fakesB += 1
            fakesB /= 2.0
            fakesA += 1
            fakesA /= 2.0
            fakesB = fakesB.transpose(0,2,3,1)
            fakesA = fakesA.transpose(0,2,3,1)

            fig_fakesA = plot_pp(fakesA)
            plt.savefig(FLAGS.pp_write + '/fakeA_' + nameB + '.png' , bbox_inches='tight')
            plt.close(fig_fakesA)

            fig_fakesB = plot_pp(fakesB)
            plt.savefig(FLAGS.pp_write + '/fakeB_' + nameA + '.png' , bbox_inches='tight')
            plt.close(fig_fakesB)

if __name__ == "__main__":
    app.run(main)
