import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import math
import time

import models

############################
#   CONSTANTS:
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_EPOCHS = 3


CODED_SIZE = 4
PATCH_SIZE = 8

PRINT_EVERY = 2000

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


# Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR LOADER
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.Dataloader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Load the model:
encoder = models.EncoderFC(CODED_SIZE, PATCH_SIZE)
decoder = models.DecoderFC(CODED_SIZE, PATCH_SIZE)

# Define the LOSS and the OPTIMIZER
criterion = nn.MSELoss()
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   TRAIN----------------------
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

num_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # Get the images
        imgs = data[0]
        # Transform the tensor into Variable
        imgs = Variable(imgs)

        # Set gradients to Zero
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        feats = encoder(imgs)
        reconstructed_imgs = decoder(feats)
        loss = criterion(reconstructed_imgs, imgs)
        loss.backward()
        optimizer.step()

        #
        running_loss += loss.data[0]
        if (i+1) % PRINT_EVERY == 0:


