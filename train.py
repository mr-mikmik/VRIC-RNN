import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import math
import time
import os
import numpy as np

import models

############################
#   CONSTANTS:
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_EPOCHS = 3


CODED_SIZE = 512
PATCH_SIZE = 32

PRINT_EVERY = 10
SAVE_STEP = 5000

MODEL_PATH = './saved_models/conv_pch32_b512/'
###########################

#==============================================
# - CUSTOM FUNCTIONS
#==============================================

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def to_patches(x, patch_size):
    num_patches_x = 32//patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            patches.append(patch.contiguous())
    return patches



# Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR LOADER
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Load the model:
encoder = models.ConvolutionalEncoder(CODED_SIZE)
decoder = models.ConvolutionalDecoder(CODED_SIZE)

# Define the LOSS and the OPTIMIZER
criterion = nn.MSELoss()
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   TRAIN----------------------
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

num_steps = len(train_loader)
start = time.time()
total_losses = []
num_patches = (32//PATCH_SIZE)**2

for epoch in range(NUM_EPOCHS):

    running_loss = 0.0
    current_losses = []
    for i, data in enumerate(train_loader, 0):

        # Get the images
        imgs = data[0]

        # Transform into patches
        patches = to_patches(imgs, PATCH_SIZE)

        for patch in patches:
            # Transform the tensor into Variable
            v_patch = Variable(patch)

            # Set gradients to Zero
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            feats = encoder(v_patch)
            reconstructed_patches = decoder(feats)
            loss = criterion(reconstructed_patches, v_patch)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        # STATISTICS:

        if (i+1) % PRINT_EVERY == 0:
            print('(%s) [%d, %5d] loss: %.3f' %
                  (timeSince(start, ((epoch * num_steps + i + 1.0) / (NUM_EPOCHS * num_steps))),
                   epoch + 1, i + 1, running_loss / PRINT_EVERY/num_patches))
            current_losses.append(running_loss/PRINT_EVERY/num_patches)
            running_loss = 0.0

        # SAVE:
        if (i + 1) % SAVE_STEP == 0:
            torch.save(decoder.state_dict(),
                       os.path.join(MODEL_PATH,
                                    'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
            torch.save(encoder.state_dict(),
                       os.path.join(MODEL_PATH,
                                    'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))
    total_losses.append(current_losses)
    torch.save(decoder.state_dict(),
               os.path.join(MODEL_PATH,
                            'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
    torch.save(encoder.state_dict(),
               os.path.join(MODEL_PATH,
                            'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))

print('__TRAINING DONE=================================================')
