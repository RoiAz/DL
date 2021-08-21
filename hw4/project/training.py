import sys
import torch
import numpy as np
import torch.optim as optim

import os
import tqdm
import IPython.display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import project.vanilla_gan as v_gan

# copyed from HW3


# optimizer
def Optimizer_func(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)

## the traing
def train_gan(v_gan,hp,data,device,gan_type_for_checkpoint_file :str):  
    print(hp)
#     num_epochs = 100
    num_epochs = 10
    
    # load params
    batch_size = hp['batch_size']
    z_dim = hp['z_dim']
    
    # set seed
    torch.manual_seed(20)

    # get data
    dl_train = DataLoader(data, batch_size, shuffle=True)

    # add model to device
    print("run model on device: ",device)
    dsc = v_gan.Discriminator(data[0][0].shape).to(device)
    gen = v_gan.Generator(z_dim, featuremap_size=4).to(device)
    
    # set optimizer function
    dsc_optimizer = Optimizer_func(dsc.parameters(), hp['discriminator_optimizer'])
    gen_optimizer = Optimizer_func(gen.parameters(), hp['generator_optimizer'])
    
    # loss funcs
    def dsc_loss_func(y_data, y_generated):
        return v_gan.discriminator_loss_fn(y_data.to(device), y_generated.to(device), hp['data_label'], hp['label_noise'])

    def gen_loss_func(y_generated):
        return v_gan.generator_loss_fn(y_generated.to(device), hp['data_label'])

    
    # get file for checkpoint
    checkpoint_file = 'checkpoints/{}'.format(gan_type_for_checkpoint_file)
    checkpoint_file_final = f'{checkpoint_file}_final'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        os.remove(f'{checkpoint_file}.pt')

    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'*** load final file {checkpoint_file_final} instead of training')
        num_epochs = 0
        gen = torch.load(f'{checkpoint_file_final}.pt', map_location=device)
        checkpoint_file = checkpoint_file_final

    # try except, if the model stop learning in the middle
    try:
        dsc_avg, gen_avg = [], []
        for epoch in range(num_epochs):
            # print batch losses
            dsc_losses, gen_losses = [], []
            print(f' EPOCH {epoch + 1}/{num_epochs} ')

            # use tqdm like we learn
            with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_data, _) in enumerate(dl_train):
                    x_data = x_data.to(device)
                    
                    # train
                    dsc_loss, gen_loss = v_gan.train_batch(
                        dsc, gen,
                        dsc_loss_func, gen_loss_func,
                        dsc_optimizer, gen_optimizer,
                        x_data)
                    
                    # append loss
                    dsc_losses.append(dsc_loss)
                    gen_losses.append(gen_loss)
                    pbar.update()

            print(f'discriminator loss - {np.mean(dsc_losses)}')
            print(f'generator loss     - {np.mean(gen_losses)}')
            dsc_avg.append(np.mean(dsc_losses))
            gen_avg.append(np.mean(gen_losses))
            
            print("444")
            # do checkpoint
            if v_gan.save_checkpoint(gen, dsc_avg, gen_avg, checkpoint_file):
                print("333")
                print(f'Saved checkpoint - {epoch + 1}/{num_epochs}')

            samples = gen.sample(5, with_grad=False)
            
    except KeyboardInterrupt as e:
        print('\n *** Training interrupted by user')
    except:
        print("Something else went wrong")

