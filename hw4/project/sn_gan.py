import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import project.v_gan as gan_vanila

# code taken from HW3 gen.py
class Discriminator(gan_vanila.Discriminator):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super(gan_vanila.Discriminator,self).__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: =====
        # params
        kernel_size = 4
        stride = 2
        padding = 1
        leaky = 0.2
        channels = [in_size[0], 128, 128, 256]

        module = []

        # create middle conv leyyers
        for i in range(len(channels)-1):
            module.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size= kernel_size, stride=stride, padding=padding, bias=False))
            module.append(nn.LeakyReLU(leaky))
            module.append(nn.BatchNorm2d(num_features=channels[i+1]))

        # Sequential it and finish it with FCL
        self.conv = nn.Sequential(*module)
        self.FCL = nn.Sequential(nn.Linear(channels[-1]*in_size[1],1))
        # ========================

# super to all the fanction we useing
class Generator(gan_vanila.Generator):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        super(Generator,self).__init__(z_dim, featuremap_size, out_channels)

def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    return gan_vanila.discriminator_loss_fn(y_data, y_generated, data_label, label_noise)

def generator_loss_fn(y_generated, data_label=0):
    return gan_vanila.generator_loss_fn(y_generated, data_label)

def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    return gan_vanila.train_batch(dsc_model, gen_model, dsc_loss_fn, gen_loss_fn, dsc_optimizer, gen_optimizer, x_data)

def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    return gan_vanila.save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file)
