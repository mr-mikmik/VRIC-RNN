import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
#import matplotlib.pyplot as plt
import scipy.misc
import numpy as np

from PIL import Image

import models
############################
#   CONSTANTS:
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_EPOCHS = 3


CODED_SIZE = 64
PATCH_SIZE = 32 

MODEL_PATH = './saved_models/'
ENCODER_PATH = './saved_models/conv_pch32_b64/encoder-3-12500.pkl'
DECODER_PATH = './saved_models/conv_pch32_b64/decoder-3-12500.pkl'

###########################


def imsave(img, name):
    img = img / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(img, './test_imgs/'+name+'.png')


def to_patches(x, patch_size):
    num_patches_x = 32//patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.contiguous())
    return patches

def reconstruct_patches(patches):
    batch_size = patches[0].size(0)
    patch_size = patches[0].size(2)
    num_patches_x = 32//patch_size
    reconstructed = torch.zeros(batch_size, 3, 32, 32)
    p = 0
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            reconstructed[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[p].data
            p += 1
    return reconstructed


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

dataiter = iter(testloader)

print('Loading Models')
# Initialize the models
encoder = models.ConvolutionalEncoder(CODED_SIZE)
decoder = models.ConvolutionalDecoder(CODED_SIZE)

# Load the SAVED model
encoder.load_state_dict(torch.load(ENCODER_PATH))
decoder.load_state_dict(torch.load(DECODER_PATH))

print('Starting eval:::::::::::::::::')
for i in range(5):
    imgs, _ = dataiter.next()
    imsave(torchvision.utils.make_grid(imgs), 'prova_'+str(i))

    # Patch the image:
    patches = to_patches(imgs, PATCH_SIZE)
    r_patches = []  # Reconstructed Patches
    for p in patches:
        feats = encoder(Variable(p))
        outputs = decoder(feats)
        r_patches.append(outputs)
    # Transform the patches into the image
    outputs = reconstruct_patches(r_patches)
    imsave(torchvision.utils.make_grid(outputs), 'prova_'+str(i)+'_decoded')
