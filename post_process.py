"""
@author: Payam Dibaeinia
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
    cycleLoss_A = 0
    nA = 0
    for a in A:
        nA += 1
        realA = a[0].cuda()
        pathA = a[2][0]
        nameA = pathA.split('/')[-1]
        nameA = nameA.split('.')[0]

        with torch.no_grad():
            model.gA.eval()
            model.gB.eval()
            fakeB = model.gA(realA)
            recycledA = model.gB(fakeB)
            cycleLoss_A += model.cycleCriterion(realA, recycledA).item()


            fakeB = fakeB.data.cpu().numpy()
            fakeB += 1
            fakeB /= 2.0
            fakeB = fakeB.transpose(0,2,3,1)

            fig_fakeB = plot_pp(fakeB)
            plt.savefig(FLAGS.pp_write + '/fakeB_' + nameA + '.png' , bbox_inches='tight')
            plt.close(fig_fakeB)

    cycleLoss_A = cycleLoss_A/nA


    cycleLoss_B = 0
    nB = 0
    for b in B:
        nB += 1
        realB = b[0].cuda()
        pathB = b[2][0]
        nameB = pathB.split('/')[-1]
        nameB = nameB.split('.')[0]

        with torch.no_grad():
            model.gA.eval()
            model.gB.eval()
            fakeA = model.gB(realB)
            recycledB = model.gA(fakeA)
            cycleLoss_B += model.cycleCriterion(realB, recycledB).item()

            fakeA = fakeA.data.cpu().numpy()
            fakeA += 1
            fakeA /= 2.0
            fakeA = fakeA.transpose(0,2,3,1)

            fig_fakeA = plot_pp(fakeA)
            plt.savefig(FLAGS.pp_write + '/fakeA_' + nameB + '.png' , bbox_inches='tight')
            plt.close(fig_fakeA)

    cycleLoss_B = cycleLoss_B/nB

    print("Forward Cycle Loss: " + str(cycleLoss_A))
    print("Backward Cycle Loss: " + str(cycleLoss_B))

if __name__ == "__main__":
    app.run(main)
