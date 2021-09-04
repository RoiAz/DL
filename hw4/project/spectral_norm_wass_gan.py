from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Based on our two implementations for spectral normalization and wass 
"""

class Discriminator(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        modules = []
        cin = in_size[0]
        cout = in_size[1] * 2
        channel_list = [cin] + [128] + [256]  + [512] + [1024]

        for ci in range(4):
            modules.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=5, stride=2, padding=2, bias=False)))
            modules.append(nn.LeakyReLU(0.1))
            cin = cout
            cout = cout *2

        self.feature_extractor = nn.Sequential(*modules)
        self.classifier = nn.Linear(4 * in_size[1] * in_size[2],1)
    

    def forward(self, x):
        feats = self.feature_extractor.forward(x).view(x.shape[0], -1)
        y = self.classifier(feats)
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.z_dim = 128

        modules = []
        cin = 1024
        cout = 1024 // 2
        self.feat_size = featuremap_size

        for ci in range(4):
            modules.append(nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=5, stride=2, padding=2, output_padding=1))
            modules.append(nn.BatchNorm2d(cout))
            modules.append(nn.ReLU())
            cin = cout
            if cin < (1024 // 4):
                modules.append(nn.ConvTranspose2d(in_channels=cin, out_channels=out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
                break
            cout = cout // 2
        modules.append(nn.Tanh())
        self.reconstructor = nn.Sequential(*modules)
        self.feats = nn.Linear(z_dim, 1024 * (featuremap_size **2))


    def sample(self, n, with_grad=False):
        device = next(self.parameters()).device
        with torch.set_grad_enabled(with_grad):
            samples = self(torch.randn(n, self.z_dim, device=device))
        return samples

    def forward(self, z):
        feats =  self.feats(z).reshape(-1, 1024, self.feat_size, self.feat_size)
        x = self.reconstructor(feats)
        return x

## we are using the dis loss and gen loss belong to w_gan        
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

#     def wass_gan_hyperparams():

#         hypers = dict(

#             data_label=1,
#             label_noise=0.0002,
#             batch_size=32,
#             z_dim=128,
            
#             discriminator_optimizer=dict(
#                 lr=0.000035 ,
#                 type='RMSprop',
#             ),
#             generator_optimizer=dict(
#                 type='RMSprop',
#                 lr=0.0001 ,
#             ),
#             N = 5 
#         )
#         return hypers
    
    def wass_gan_hyperparams():   
    
        hypers = dict(
        
            batch_size=32,
            z_dim=128,
            data_label=1,
            label_noise=0.2,
            discriminator_optimizer=dict(
                type="SGD",
                lr=0.0075 ,
            ),
            generator_optimizer=dict(
                type="Adam",
                lr= 0.001,
                betas =(0.5, 0.999),
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

#using our vanilla checkpoint
def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"
    from statistics import mean
    early_stopping = False
    if len(dsc_losses) > 3 and mean([dsc_losses[-2], dsc_losses[-3], dsc_losses[-3]]) > dsc_losses[-1] and \
    len(gen_losses) > 3 and mean([gen_losses[-2], gen_losses[-3], gen_losses[-3]]) > gen_losses[-1]:
        early_stopping = True
    
    if early_stopping:
        torch.save(gen_model, checkpoint_file)
        saved = True

    return saved