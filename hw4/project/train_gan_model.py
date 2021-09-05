import cs236781.plot as plot
import cs236781.download
import re
import zipfile
import unittest
import urllib
import shutil
import os
import sys
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch.nn.Upsample as upsample 
from .score_inception import inception_score
reset = False


"""
Training model using two different function, for model-train and gan-train
Based on the implementation idea of hw3
"""

# hyperparams
def v_gan_hyperparams():
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
    )
    return hypers

def sn_gan_hyperparams():
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
    )
    return hypers

def w_gan_hyperparams():
    hypers = dict(
        batch_size=32,
        z_dim=128,
        data_label=1,
        label_noise=0.0002,
        discriminator_optimizer=dict(
            type='RMSprop',
            lr=0.000035 ,
        ),
        generator_optimizer=dict(
            type='RMSprop',
            lr=0.0001 ,
        ),
        N = 5 
    )
    return hypers

def train_gan_model(device, ds_gwb, modelCodeModule, checkpoint_file_suffix : str, hyperparams:dict):
    print(type(device))
    print(hyperparams)
    print(modelCodeModule)
    imageDir = os.path.join('project/generated_imgs', checkpoint_file_suffix)
    if not os.path.exists(imageDir):
        os.makedirs(imageDir)
    
    imageSrc = os.path.join(imageDir,'{}_sample_images.pt'.format(checkpoint_file_suffix))
    gen = None
    if os.path.exists(imageSrc) and not reset:
        samples = torch.load(imageSrc)
    else:
        checkpoint_file = 'checkpoints/'+checkpoint_file_suffix
        if not os.path.isfile(f'{checkpoint_file}.pt'):
            train_gan(modelCodeModule, hyperparams, ds_gwb, device, checkpoint_file_suffix)
#             print("train_model_111")
        if os.path.isfile(f'{checkpoint_file}.pt'):
#             print("train_model_222")
            gen = torch.load(f'{checkpoint_file}.pt', map_location=device)
            print(gen)
        samples = gen.sample(n=530, with_grad=False).cpu()
        torch.save(samples,imageSrc)
        

    
    # plot images
#     print('*** Images Generated from model of the {}:'.format(checkpoint_file_suffix))
#     fig, _ = plot.tensors_as_images(samples, nrows=1, figsize=(20,20))
    
    
    
    # inception_score
#     print(f'The size of samples:{type(samples)}')
    

    """    
    # 3x299x299
    self.Upsample = upsample(size=[299,299], mode=‘bilinear’)
    samples = torch.unsqueeze(samples, 0) #to get 1×C×H×W
    samples = self.upsample(samples)

    """
#     temp = gen.sample(n=530, with_grad=False)
#     print(samples.shape)
#     samples = torch.unsqueeze(samples, 0) #to get 1×C×H×W
#     curr_inception_score, _ = inception_score(samples,cuda=True,batch_size=64, resize=True)
#     print(f'Inception Score: {curr_inception_score}')
#     mean , scores = inception_score(samples, cuda=True, batch_size=32, resize=False, splits=1)
#     print(f'Inception score of {checkpoint_file_suffix}: scores are: {scores} and mean score is: {mean}.')
    
    #fig.savefig(imageSrc)
    
    return gen

def train_gan(v_gan,hp,data,device,gan_type_for_checkpoint_file :str):
    
    def Optimizer_func(model_params, opt_params):
        opt_params = opt_params.copy()
        optimizer_type = opt_params['type']
        opt_params.pop('type')
        return optim.__dict__[optimizer_type](model_params, **opt_params)
    
#     print(hp)
    num_epochs = 300
    
    # load params
    batch_size = hp['batch_size']
    z_dim = hp['z_dim']
    
    # set seed
    torch.manual_seed(20)

    # get data
#     dl_train = DataLoader(data, batch_size, shuffle=True)
    dl_train = torch.utils.data.DataLoader(data, batch_size=batch_size)

#     inception score
#     mean , scores = inception_score(dl_train, cuda=True, batch_size=32, resize=False, splits=1)
#     print(f'$$$$$$$$$$$$$$$ scores are {scores} and mean score is{mean}.')
    
    # add model to device
#     print("run model on device: ",device)
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
#         print(f'*** load final file {checkpoint_file_final} instead of training')
        num_epochs = 0
        gen = torch.load(f'{checkpoint_file_final}.pt', map_location=device)
        checkpoint_file = checkpoint_file_final

    # try except, if the model stop learning in the middle
    try:
        dsc_avg, gen_avg, inception_lst = [], [], []
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
            
            # inception list maintainance
            samples = gen.sample(530, with_grad=False)
            np_inception_score, _ = inception_score(samples,cuda=True,batch_size=64, resize=True)
            
            # debbug
         #   print(f'inception shape: {np_inception_score.shape}')
          #  print(f'inception type: {type(np_inception_score)}')
            
            int_inception_score = np_inception_score.tolist()
            inception_lst.append(int_inception_score)
            
            
        #    print("444")
            # do checkpoint
            if v_gan.save_checkpoint(gen, dsc_avg, gen_avg, checkpoint_file):
#                 print("333")
                print(f'Saved checkpoint - {epoch + 1}/{num_epochs}')

#             samples = gen.sample(5, with_grad=False)
        
        # saves inception results for graph creation
        name = "inception_%s" % gan_type_for_checkpoint_file
        filename = "%s.txt" % name
        path = "project/outputs_for_graphs/"
        file = open(path + filename, "w") 
        inception_listToStr = ' '.join([str(elem) for elem in inception_lst])
#         inception_listToStr = ''.join([str(elem)+"\n" for elem in inception_lst])
        file.write(''.join(inception_listToStr)) 
        file.close()
        
        # saves discriminator avg loss results for graph creation
        name = "discriminator_%s" % gan_type_for_checkpoint_file
        filename = "%s.txt" % name
        path = "project/outputs_for_graphs/"
        file = open(path + filename, "w") 
        dis_listToStr = ' '.join([str(elem) for elem in dsc_avg])
#         dis_listToStr = ''.join([str(elem)+"\n" for elem in dsc_avg])
        file.write(''.join(dis_listToStr)) 
        file.close()
        
        # saves generator avg loss results for graph creation
        name = "generator_%s" % gan_type_for_checkpoint_file
        filename = "%s.txt" % name
        path = "project/outputs_for_graphs/"
        file = open(path + filename, "w") 
        gen_listToStr = ' '.join([str(elem) for elem in gen_avg])
#         gen_listToStr = ''.join([str(elem)+"\n" for elem in gen_avg])
        file.write(''.join(gen_listToStr))
        file.close()
        
    
    except KeyboardInterrupt as e:
        print('\n *** Training interrupted by user')
#     except:
#         print("Something else went wrong")