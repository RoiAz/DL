from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import project.vanilla_gan as gan_vanila
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

""" Code is based on the paper reference(we got here): 
    https://arxiv.org/pdf/1701.07875.pdf
    and with the help of this youtube video:
    https://www.youtube.com/watch?v=pG0QZ7OddX4
                                                """

class Generator(gan_vanila.Generator):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        super(Generator,self).__init__(z_dim, featuremap_size, out_channels)

class Discriminator(gan_vanila.Discriminator):
    def __init__(self, in_size):
        super(Discriminator,self).__init__(in_size)

def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    assert data_label == 1 or data_label == 0
    return -(torch.mean(y_data) - torch.mean(y_generated))

def generator_loss_fn(y_generated, data_label=0):
    assert data_label == 1 or data_label == 0
    loss = -torch.mean(y_generated)
    return loss

def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):

    def wass_gan_hyperparams():
        hypers = dict(

            data_label=1,
            label_noise=0.0002,
            batch_size=32,
            z_dim=128,
            
            discriminator_optimizer=dict(
                lr=0.000035 ,
                type='RMSprop',
            ),
            generator_optimizer=dict(
                type='RMSprop',
                lr=0.0001 ,
            ),
            N = 5 
        )
        return hypers

    num = wass_gan_hyperparams()['N']
    
    for i in range(num):
        dsc_optimizer.zero_grad()

        data = gen_model.sample(x_data.shape[0], with_grad=True)
        previous_generated = dsc_model(data.detach())
        previous_data = dsc_model(x_data)
        dsc_loss = dsc_loss_fn(previous_data, previous_generated)
        dsc_loss.backward()
        dsc_optimizer.step()

        for p in dsc_model.parameters():
            p.data.clamp_(-0.01, 0.01)

    gen_optimizer.zero_grad()
    sampels = gen_model.sample(x_data.shape[0], with_grad=True)
    gen_label = dsc_model(sampels)
    gen_loss = gen_loss_fn(gen_label)
    gen_loss.backward()
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ====== 
    from statistics import mean
    early_stopping = False
    if len(dsc_losses) > 3 and mean([dsc_losses[-2], dsc_losses[-3], dsc_losses[-3]]) > dsc_losses[-1] and \
    len(gen_losses) > 3 and mean([gen_losses[-2], gen_losses[-3], gen_losses[-3]]) > gen_losses[-1]:
        early_stopping = True
    
    if early_stopping:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved