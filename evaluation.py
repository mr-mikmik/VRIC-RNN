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


CODED_SIZE = 16
PATCH_SIZE = 8

PRINT_EVERY = 2000
SAVE_STEP = 5000

MODEL_PATH = './saved_models/'
ENCODER_PATH = './saved_models/encoder-3-12500.pkl'
DECODER_PATH = './saved_models/decoder-3-12500.pkl'

###########################


def imsave(img, name):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    im = Image.fromarray(npimg)
    im.save('./test_imgs/'+name, np.transpose(npimg, (1, 2, 0)))


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
encoder = models.EncoderFC(CODED_SIZE, PATCH_SIZE)
decoder = models.DecoderFC(CODED_SIZE, PATCH_SIZE)

# Load the SAVED model
encoder.load_state_dict(torch.load(ENCODER_PATH))
decoder.load_state_dict(torch.load(DECODER_PATH))

print('Starting eval:::::::::::::::::')
for i in range(5):
    imgs, _ = dataiter.next()
    imsave(torchvision.utils.make_grid(imgs), 'prova1.jpg')
    feats = encoder(Variable(imgs))
    outputs = decoder(feats)
    imsave(torchvision.utils.make_grid(torch.Tensor(outputs.data)), 'prova1_decoded.jpg')
