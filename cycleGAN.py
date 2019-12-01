"""
@author: Payam Dibaeinia, and {add your names}
"""

import torch
import torch.nn as nn
from generator import generator
from discriminator import discriminator
from buffer import buffer
import itertools
from utils import plot, init_weights
import matplotlib.pyplot as plt



class cycleGAN(object):
    """
    generatorA : realA --> fakeB
    discriminatorA: fakeB vs realB
    generatorB: realB --> fakeA
    discriminatorB: fakeA vs realA
    """

    def __init__(self, flags):

        self.nIter_ = 0
        self.avgLoss = {
            'dA' : 0,
            'dB' : 0,
            'gan_gA' : 0,
            'gan_gB' : 0,
            'cycle_forw' : 0,
            'cycle_back' : 0,
            'idnt_gA' : 0,
            'idnt_gB' : 0,
        }
        self.flags_ = flags
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ganLossType_ = flags.glt

        # Identity loss can be used only if the A and B have the same number channels
        if flags.l_idnt > 0:
            assert(flags.ncA == flags.ncB)

        self.gA = generator(in_channels = flags.ncA, out_channels = flags.ncB, nBlocks = flags.nbl, nChanFirstConv = flags.ncFirstConv, dropout = flags.dropout)
        self.gB = generator(in_channels = flags.ncB, out_channels = flags.ncA, nBlocks = flags.nbl, nChanFirstConv = flags.ncFirstConv, dropout = flags.dropout)
        self.dA = discriminator(in_channels = flags.ncB, nChanFirstConv = flags.ncFirstConv)
        self.dB = discriminator(in_channels = flags.ncA, nChanFirstConv = flags.ncFirstConv)

        self.gA.to(self.device_)
        self.gB.to(self.device_)
        self.dA.to(self.device_)
        self.dB.to(self.device_)

        init_weights(self.gA, flags.init_type, flags.init_scale)
        init_weights(self.gB, flags.init_type, flags.init_scale)
        init_weights(self.dA, flags.init_type, flags.init_scale)
        init_weights(self.dB, flags.init_type, flags.init_scale)

        self.buffer_fakeA = buffer()
        self.buffer_fakeB = buffer()

        #TODO 'was' mode of gan loss needs implementation of gradient penalty as well

        if self.ganLossType_ == 'lse': #Least square loss
            self.ganCriterion = nn.MSELoss()
        elif self.ganLossType_ == 'bce': #binary cross entropy loss i.e. binary classification
            self.ganCriterion =  nn.BCEWithLogitsLoss()
        elif self.ganLossType_ == 'was': #Wasserstein GAN
            self.ganCriterion = None
        else:
            raise NotImplementedError('Specified gan loss has not been implemented')


        self.cycleCriterion = nn.L1Loss()
        self.idntCriterion = nn.L1Loss()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.gA.parameters(), self.gB.parameters()), lr = flags.lr, betas = (0, 0.999))
        self.optimizerD = torch.optim.Adam(itertools.chain(self.dA.parameters(), self.dB.parameters()), lr = flags.lr, betas = (0, 0.999))

    def forward(self, input):
        """ calculates forward pass and set variables

        input: a tuple of tensors (A,B)
        """
        self.realA = input[0]
        self.realB = input[1]
        self.fakeB = self.gA(self.realA)
        self.fakeA = self.gB(self.realB)
        self.recycledA = self.gB(self.fakeB)
        self.recycledB = self.gA(self.fakeA)

    def _gan_lossD(self, disc, realImage, fakeImage):
        """Helper function to calculate the two terms in the loss function when training discriminator, i.e.:
        ganCriterion(disc(real), True) + ganCriterion(disc(fake), False)

        disc: discriminator network
        realImage: the real image (either A or B)
        fakeImage: the output of generator (either gA, or gB)
        """
        predReal = disc(realImage)
        targetReal = torch.tensor(1.0).expand_as(predReal)
        targetReal = targetReal.to(self.device_)
        lossReal = self.ganCriterion(predReal, targetReal)

        predFake = disc(fakeImage.detach())
        targetFake = torch.tensor(0.0).expand_as(predFake)
        targetFake = targetFake.to(self.device_)
        lossFake = self.ganCriterion(predFake, targetFake)

        loss = 0.5 * (lossReal + lossFake) # don't do backward here
        return loss

    def _gan_idnt_lossG(self, disc, gen, realImage, fakeImage):
        """Helper function to calculate gan and identity loss for training generator, i.e.:
        ganCriterion(disc(fake), True) + idntCriterion(gen(real), real)

        disc: discriminator network
        gen: generator network
        realImage: the real image (either A or B)
        fakeImage: the output of generator (either gA, or gB)

        returns lossGAN, lossIdnt, identity
        """
        # calculate GAN loss
        pred = disc(fakeImage)
        target = torch.tensor(1.0).expand_as(pred)
        target = target.to(self.device_)
        lossGAN = self.ganCriterion(pred, target)

        # calculate identity loss
        lossIdnt = 0
        identity = None
        if self.flags_.l_idnt > 0:
            identity = gen(realImage)
            lossIdnt = self.idntCriterion(identity, realImage)

        return lossGAN, lossIdnt, identity

    def backwardD(self):
        """
        Calculates backward pass for updating discriminators (both dA and dB)
        """
        # loss and backward for dA
        fakeB = self.buffer_fakeB.getImages(self.fakeB)
        fakeB = fakeB.to(self.device_)
        self.loss_dA = self._gan_lossD(self.dA, self.realB, fakeB)
        #self.loss_dA.backward()

        # backward for dB
        fakeA = self.buffer_fakeA.getImages(self.fakeA)
        fakeA = fakeA.to(self.device_)
        self.loss_dB = self._gan_lossD(self.dB, self.realA, fakeA)
        #self.loss_dB.backward()
        self.lossD = self.loss_dA + self.loss_dB
        self.lossD.backward()

    def backwardG(self):
        """
        Calculates backward pass for updating generators (both gA and gB)
        """
        lambdaA = self.flags_.l_A
        lambdaB = self.flags_.l_B
        lambdaIdnt = self.flags_.l_idnt
        if not self.ganLossType_ == 'was':
            # GAN and identity loss for gA
            self.loss_gan_gA, self.loss_idnt_gA, self.idntA = self._gan_idnt_lossG(self.dA, self.gA, self.realB, self.fakeB)
            self.loss_idnt_gA *= lambdaB * lambdaIdnt

            # GAN and identity loss for gB
            self.loss_gan_gB, self.loss_idnt_gB, self.idntB = self._gan_idnt_lossG(self.dB, self.gB, self.realA, self.fakeA)
            self.loss_idnt_gB *= lambdaA * lambdaIdnt
        else:
            # 'was' loss is not imlemented yet, it changes both helper backward functions
            # cause ganCriterion is not defined anymore. Also it needs gradient penalty loss
            # TODO: if implement Wgan, 1- modify both backward helper functions 2- move this if condition
            # to the helper functions 3- gradient penalty
            raise NotImplementedError('Specified gan loss has not been implemented')

        # forward cycle loss
        self.loss_cycle_forw = self.cycleCriterion(self.recycledA, self.realA) * lambdaA
        # backward cycle loss
        self.loss_cycle_back = self.cycleCriterion(self.recycledB, self.realB) * lambdaB

        self.lossG = self.loss_gan_gA + self.loss_gan_gB + self.loss_idnt_gA + self.loss_idnt_gB + self.loss_cycle_forw + self.loss_cycle_back
        self.lossG.backward()

    def update_optimizer(self, input):
        """
        Runs forward pass, then optimizes generators and updates their parameters,
        then optimizes discriminators and updates their parameters.
        """
        A,B = input
        A = A.to(self.device_)
        B = B.to(self.device_)
        input = (A,B)

        self.gA.train()
        self.gB.train()
        self.dA.train()
        self.dB.train()


        self.forward(input)
        for param in self.dA.parameters():
            param.requires_grad = False
        for param in self.dB.parameters():
            param.requires_grad = False

        self.optimizerG.zero_grad()
        self.backwardG()
        self.optimizerG.step()

        for param in self.dA.parameters():
            param.requires_grad = True
        for param in self.dB.parameters():
            param.requires_grad = True

        self.optimizerD.zero_grad()
        self.backwardD()
        self.optimizerD.step()

        self.nIter_ += 1
        self._update_loss_dict()
    def save(self, path, epoch):
        """
        path e.g. /saved_models/
        """
        torch.save({
            'epoch': epoch,
            'gA_state_dict': self.gA.state_dict(),
            'gB_state_dict': self.gB.state_dict(),
            'dA_state_dict': self.dA.state_dict(),
            'dB_state_dict': self.dB.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'buffer_fakeA': self.buffer_fakeA,
            'buffer_fakeB': self.buffer_fakeB,
            'nIter': self.nIter_,
            'loss_dict': self.avgLoss,
            }, path + '/checkpoint_' + str(epoch) + '.tar')

    def load(self, path):
        checkpoint = torch.load(path)
        self.gA.load_state_dict(checkpoint['gA_state_dict'])
        self.gB.load_state_dict(checkpoint['gB_state_dict'])
        self.dA.load_state_dict(checkpoint['dA_state_dict'])
        self.dB.load_state_dict(checkpoint['dB_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.buffer_fakeA = checkpoint['buffer_fakeA']
        self.buffer_fakeB = checkpoint['buffer_fakeB']
        self.nIter_ = checkpoint['nIter']
        self.avgLoss = checkpoint['loss_dict']

        return checkpoint['epoch']

    def evaluate(self, imagesA, imagesB, path_write, epoch):
        """
        Samples from the generator are scaled between -1 and 1.
        The plot function expects them to be scaled between 0 and 1 and also
        expects the order of the channels to be (batch_size,w,h,3) as opposed to
        how PyTorch expects it.

        path_write e.g. /generated_images/train/ or /generated_images/test/

        #DEBUG: make sure this scaling and our plot function does not cause any problem
        """
        imagesA = imagesA.to(self.device_)
        imagesB = imagesB.to(self.device_)

        with torch.no_grad():
            self.gA.eval()
            self.gB.eval()
            fakesB = self.gA(imagesA)
            fakesA = self.gB(imagesB)

            fakesB = fakesB.data.cpu().numpy()
            fakesA = fakesA.data.cpu().numpy()
            fakesB += 1
            fakesB /= 2.0
            fakesA += 1
            fakesA /= 2.0
            fakesB = fakesB.transpose(0,2,3,1)
            fakesA = fakesA.transpose(0,2,3,1)

            fig_fakesA = plot(fakesA)
            plt.savefig(path_write + '/fakeA_' + str(epoch), bbox_inches='tight')
            plt.close(fig_fakesA)

            fig_fakesB = plot(fakesB)
            plt.savefig(path_write + '/fakeB_' + str(epoch), bbox_inches='tight')
            plt.close(fig_fakesB)

    def update_lr(self, newLR_G, newLR_D):
        """
        Updates the learn rate for generator and discriminator optimizers
        """
        for param_group in self.optimizerG.param_groups:
            param_group['lr'] = newLR_G

        for param_group in self.optimizerD.param_groups:
            param_group['lr'] = newLR_D

    def print_loss(self, epoch):
        """
        prints the values of all 8 losses in the following order:
        epoch, loss_dA, loss_dB, loss_gan_gA, loss_gan_gB, loss_cycle_forw, loss_cycle_back, loss_idnt_gA, loss_idnt_gB
        """
        print(epoch, "%.3f" % self.avgLoss['dA'],
                        "%.3f" % self.avgLoss['dB'],
                        "%.3f" % self.avgLoss['gan_gA'],
                        "%.3f" % self.avgLoss['gan_gB'],
                        "%.3f" % self.avgLoss['cycle_forw'],
                        "%.3f" % self.avgLoss['cycle_back'],
                        "%.3f" % self.avgLoss['idnt_gA'],
                        "%.3f" % self.avgLoss['idnt_gB'])

    def _update_loss_dict(self):
        N = self.nIter_
        self.avgLoss['dA'] = ((N-1) * self.avgLoss['dA'] + self.loss_dA)/N
        self.avgLoss['dB'] = ((N-1) * self.avgLoss['dB'] + self.loss_dB)/N
        self.avgLoss['gan_gA'] = ((N-1) * self.avgLoss['gan_gA'] + self.loss_gan_gA)/N
        self.avgLoss['gan_gB'] = ((N-1) * self.avgLoss['gan_gB'] + self.loss_gan_gB)/N
        self.avgLoss['cycle_forw'] = ((N-1) * self.avgLoss['cycle_forw'] + self.loss_cycle_forw)/N
        self.avgLoss['cycle_back'] = ((N-1) * self.avgLoss['cycle_back'] + self.loss_cycle_back)/N
        self.avgLoss['idnt_gA'] = ((N-1) * self.avgLoss['idnt_gA'] + self.loss_idnt_gA)/N
        self.avgLoss['idnt_gB'] = ((N-1) * self.avgLoss['idnt_gB'] + self.loss_idnt_gB)/N

    # TODO: 1- add BW specific command for ADAM opt
