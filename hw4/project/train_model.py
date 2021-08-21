import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re
import zipfile

import numpy as np
import torch
import matplotlib.pyplot as plt

import cs236781.plot as plot
import cs236781.download

from project.training import train_gan

# help function, mustly copyed from HW3
# reset model if we yet to made one
reset = False

def train_model(device, ds_gwb, modelCodeModule, checkpoint_file_suffix : str, hyperparams:dict):
    print(hyperparams)
    print(modelCodeModule)
    imageDir = os.path.join('project_imgs', checkpoint_file_suffix)
    # create dir
    if not os.path.exists(imageDir):
        os.makedirs(imageDir)
    
    # lornd from the model only if needed
    imageSrc = os.path.join(imageDir,'{}_sample_images.pt'.format(checkpoint_file_suffix))
    gen = None
    if os.path.exists(imageSrc) and not reset:
        samples = torch.load(imageSrc)
    else:
        checkpoint_file = 'checkpoints/'+checkpoint_file_suffix
        if not os.path.isfile(f'{checkpoint_file}.pt'):
            train_gan(modelCodeModule, hyperparams, ds_gwb, device, checkpoint_file_suffix)
            print("111")
        if os.path.isfile(f'{checkpoint_file}.pt'):
            print("222")
            gen = torch.load(f'{checkpoint_file}.pt', map_location=device)
            print(gen)
        samples = gen.sample(n=10, with_grad=False).cpu()
        torch.save(samples,imageSrc)
    
    # plot images
    print('*** Images Generated from model of the {}:'.format(checkpoint_file_suffix))
    fig, _ = plot.tensors_as_images(samples, nrows=1, figsize=(20,20))
    #fig.savefig(imageSrc)
    
    return gen