"""
@author: Payam Dibaeinia, and {add your names}
"""

from absl import flags
from absl import logging
from absl import app
from data_tools import buildDataLoader
from utils import make_dir
from cycleGAN import cycleGAN
from utils import calculate_lr


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




def main(argv):

    if not FLAGS.resume:
        make_dir(flags)
    else:
        if not FLAGS.last_epoch:
            raise RuntimeError('Specify the checkpoint from which you want to resume training')
        else:
            path_checkpoint = FLAGS.checkpoint_dir + '/checkpoint_' + str(FLAGS.last_epoch) + '.tar'

    """
    Prepare data
    """
    data = buildDataLoader(FLAGS.ds, FLAGS.data_dir)
    data.sampleData(FLAGS.nSamples, indices = None)

    #TODO take care of transformation

    train_data = data.getDataLoader_train(batch_size = FLAGS.bs, transform = None, nWorkers = FLAGS.nw)
    trainLoaderA, trainLoaderB = train_data
    #Commented test data for now as it won't be used during training, instead we work on a sub-sample of test data
    #test_data = data.getDataLoader_test(batch_size = FLAGS.bs, transform = None, nWorkers = FLAGS.nw)
    #testLoaderA, testLoaderB = test_data

    #TODO take care of transformation
    trainA_samples, trainB_samples, testA_samples, testB_samples = data.getDataLoader_samples(transform_train = None, transform_test = None, nWorkers = FLAGS.nw)

    """
    Define model
    """
    last_epoch = -1
    model = cycleGAN(FLAGS)
    if FLAGS.resume:
        last_epoch = model.load(path_checkpoint)

    if not last_epoch == FLAGS.last_epoch:
        raise RuntimeError('DEBUG: Inconsistency between the saved last epoch and specified last epoch')

    """
    Training
    """
    for epoch in range(last_epoch+1, FLAGS.nEpoch):

        ##### BW specific command for ADAM opt #######
        for group in model.optimizerD.param_groups:
            for p in group['params']:
                state = model.optimizerD.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        for group in model.optimizerG.param_groups:
            for p in group['params']:
                state = model.optimizerG.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        ################################################


        #NOTE: I think iterator on dataloader by default returns a list where first element is data and second is its label
        # but here we don't have label, therefore we only need the first element, so index '0' was used
        for A, B in zip(trainLoaderA, trainLoaderB):
            input = (A[0],B[0])
            model.update_optimizer(input)
            model.print_loss(epoch)

        """
        Save and Evaluation
        """
        if (epoch + 1) % FLAGS.freq == 0:
            model.save(FLAGS.checkpoint_dir, epoch)

            for sampledA, sampledB in zip(trainA_samples, trainB_samples):
                path_write = flags.image_dir + '/train'
                model.evaluate(sampledA[0], sampledB[0], path_write, epoch)

            for sampledA, sampledB in zip(testA_samples, testB_samples):
                path_write = flags.image_dir + '/test'
                model.evaluate(sampledA[0], sampledB[0], path_write, epoch)

        """
        Update learning rate
        """
        newLR = calculate_lr(FLAGS.lr, FLAGS.steps_constLR, FLAGS.nEpoch - FLAGS.steps_constLR, self.nEpoch, epoch)
        model.update_lr(newLR_G = newLR, newLR_D = newLR)
