"""
@author: Payam Dibaeinia, and {add your names}
"""

import torch
import torch.nn as nn
from generator import generator
from discriminator import discriminator
from buffer import buffer
import itertools
from utils import plot
import matplotlib.pyplot as plt



class cycleGAN(object):
    """
    generatorA : realA --> fakeB
    discriminatorA: fakeB vs realB
    generatorB: realB --> fakeA
    discriminatorB: fakeA vs realA
    """

    def __init__(self, flags):

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
        self.realA = input[0].to(self.device_)
        self.realB = input[1].to(self.device_)
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
        targetReal = torch.tensor(True).expand_as(predReal)
        lossReal = self.ganCriterion(predReal, targetReal)

        predFake = disc(fakeImage)
        targetFake = torch.tensor(False).expand_as(predFake)
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
        target = torch.tensor(True).expand_as(pred)
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
        self.loss_dA = self._gan_lossD(self.dA, self.realB, fakeB)
        #self.loss_dA.backward()

        # backward for dB
        fakeA = self.buffer_fakeA.getImages(self.fakeA)
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
        if not self.flags_.ganLossType_ == 'was':
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
            }, path + '/checkpoint_' + str(epoch) + '.tar')

    def load(self, path):
        checkpoint = torch.load(path)
        self.gA.load_state_dict(checkpoint['gA_state_dict'])
        self.gB.load_state_dict(checkpoint['gB_state_dict'])
        self.dA.load_state_dict(checkpoint['dA_state_dict'])
        self.dB.load_state_dict(checkpoint['dB_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.buffer_fakeA = load_state_dict(checkpoint['buffer_fakeA'])
        self.buffer_fakeB = load_state_dict(checkpoint['buffer_fakeB'])

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
        print(epoch, "%.3f" % self.loss_dA,
                        "%.3f" % self.loss_dB,
                        "%.3f" % self.loss_gan_gA,
                        "%.3f" % self.loss_gan_gB,
                        "%.3f" % self.loss_cycle_forw,
                        "%.3f" % self.loss_cycle_back,
                        "%.3f" % self.loss_idnt_gA,
                        "%.3f" % self.loss_idnt_gB)

    # TODO: 1- add BW specific command for ADAM opt
