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
import torch.distributed as dist

class buildDataLoader(object):
    """
    Create multi-threaded pytorch data loaders
    """

    def __init__(self, dataset_name, path_data):
        """
        dataset_name: str; 'horse2zebra', 'apple2orange', 'summer2winter_yosemite', ...
        path_data: directory of data folder where datasets exist or hould be downloaded to
        """


        self.name_ = dataset_name
        #TODO: complete this
        self.url_dict_ = {
            'horse2zebra':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
            'apple2orange':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip',
            'summer2winter_yosemite':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip',
            'maps':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip',
            'cezanne2photo':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cezanne2photo.zip',
            'monet2photo':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip',
            'ukiyoe2photo':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip',
            'vangogh2photo':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip',
            'iphone2dslr_flower':'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/iphone2dslr_flower.zip',
        }
        if not self._check_exists(path_data):
            self._download(path_data)
            self._unzip(path_data + '/' + self.name_ + '.zip', path_data)
            self._orginize_folders(path_data + '/' + self.name_)

        self.data_path_ = path_data + '/' + self.name_

    def _check_exists(self, path):
        currPath = path+ '/' + self.name_
        return os.path.exists(currPath)

    def _download(self, path_download):
        url = self.url_dict_[self.name_]
        utils.download_url(url, root=path_download, filename = self.name_ + '.zip', md5 = None)

    def _unzip(self, path_zip, path_extract):
        with ZipFile(path_zip, 'r') as zipObj:
           zipObj.extractall(path = path_extract)
        os.remove(path_zip)

    def _orginize_folders(self, path_dataset):
        """
        In each subdirectory of the dataset (i.e. trainA, trainB, ...), creates
        a new folder named '1', and transfers all images to this new folder.
        This is required for creating a pytorch dataloader.
        """
        fnames = ['trainA','trainB','testA','testB']
        for f in fnames:
            currPath = path_dataset + '/' + f
            os.mkdir(currPath + '/1')

            for src in os.listdir(currPath):
                if src != '1' and src != '.DS_Store':
                    shutil.copy(currPath + '/' + src, currPath + '/1')
                    os.remove(currPath + '/' + src)

    def getDataLoader_train(self, batch_size, transform = None, nWorkers = 1):
        """returns pytorch dataloader with the specified batch size, transformation and thread

        transformation: a torchvision transformation, if None only tensor conversion is applied
        #TODO: make sure if in cycleGAN paper any transformation such as normalization was used
        """
        if not transform:
            transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        datasetA = datasets.ImageFolder(self.data_path_ + '/trainA/', transform = transform)
        datasetB = datasets.ImageFolder(self.data_path_ + '/trainB/', transform = transform)

        trainLoaderA = torch.utils.data.DataLoader(datasetA, batch_size = batch_size, shuffle = True, num_workers = nWorkers)
        trainLoaderB = torch.utils.data.DataLoader(datasetB, batch_size = batch_size, shuffle = True, num_workers = nWorkers)

        return trainLoaderA, trainLoaderB

    def getDataLoader_test(self, batch_size, transform = None, nWorkers = 1):
        """returns pytorch dataloader with the specified batch size, transformation and thread

        transformation: a torchvision transformation, if None only tensor conversion is applied
        #TODO: make sure if in cycleGAN paper any transformation such as normalization was used
        """
        if not transform:
            transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        datasetA = datasets.ImageFolder(self.data_path_ + '/testA/', transform = transform)
        datasetB = datasets.ImageFolder(self.data_path_ + '/testB/', transform = transform)

        testLoaderA = torch.utils.data.DataLoader(datasetA, batch_size = batch_size, shuffle = False, num_workers = nWorkers)
        testLoaderB = torch.utils.data.DataLoader(datasetB, batch_size = batch_size, shuffle = False, num_workers = nWorkers)

        return testLoaderA, testLoaderB

    def sampleData(self, nSamples, indices = None):
        """samples some data from all four datasets (trainA, testA, trainB, testB)
        that will be used for evaluation during training

        indices: if given, they used for  sampling else randpm indices are generated.
        Indices refers to indices when os.listdir() is used
        """

        self._sample_data(nSamples, 'trainA', indices = indices)
        self._sample_data(nSamples, 'trainB', indices = indices)
        self._sample_data(nSamples, 'testA', indices = indices)
        self._sample_data(nSamples, 'testB', indices = indices)

    def getDataLoader_samples(self, transform_train = None, transform_test = None, nWorkers = 1):
        """returns pytorch dataloader for the sampled data.

        transformation: a torchvision transformation, if None only tensor conversion is applied
        #TODO: make sure if in cycleGAN paper any transformation such as normalization was used
        """
        if not transform_train:
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])

        if not transform_test:
            transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])


        datasetA_train = datasets.ImageFolder(self.data_path_ + '/sample_trainA/', transform = transform_train)
        datasetA_test = datasets.ImageFolder(self.data_path_ + '/sample_testA/', transform = transform_test)
        datasetB_train = datasets.ImageFolder(self.data_path_ + '/sample_trainB/', transform = transform_train)
        datasetB_test = datasets.ImageFolder(self.data_path_ + '/sample_testB/', transform = transform_test)

        trainLoaderA = torch.utils.data.DataLoader(datasetA_train, batch_size = len(datasetA_train), shuffle = False, num_workers = nWorkers)
        testLoaderA = torch.utils.data.DataLoader(datasetA_test, batch_size = len(datasetA_test), shuffle = False, num_workers = nWorkers)
        trainLoaderB = torch.utils.data.DataLoader(datasetB_train, batch_size = len(datasetB_train), shuffle = False, num_workers = nWorkers)
        testLoaderB = torch.utils.data.DataLoader(datasetB_test, batch_size = len(datasetB_test), shuffle = False, num_workers = nWorkers)

        return trainLoaderA, trainLoaderB, testLoaderA, testLoaderB

    def _sample_data(self, nSamples, folder_name, indices = None):
        """helper function for sampling some data from the given folder name (e.g. trainA)
        that will be used for evaluation during training

        indices: if given, they used for  sampling else randpm indices are generated.
        Indices refers to indices when os.listdir() is used
        """
        path = self.data_path_ + '/sample_' + folder_name
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path + '/1')
        else:
            shutil.rmtree(path + '/1')
            os.mkdir(path + '/1')

        originalPath = self.data_path_ + '/' + folder_name + '/1'
        allImages =  os.listdir(originalPath)
        try:
            allImages.remove('.DS_Store')
        except:
            pass


        if not indices:
            indices = np.random.randint(0, len(allImages), nSamples)
        fnames = np.take(allImages, indices)
        for f in fnames:
            src = originalPath + '/' + f
            dst = self.data_path_ + '/sample_' + folder_name + '/1'
            shutil.copy(src, dst)
