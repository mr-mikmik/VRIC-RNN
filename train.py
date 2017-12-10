import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
import math
import time
import os
import numpy as np

import models


def main(args):

    # Create the model directory if does not exist
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR LOADER
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Load the model:
    model = models.setup(args)

    # Define the LOSS and the OPTIMIZER
    criterion = nn.MSELoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # ::::::::::::::::::::::::::::::::
    #   TRAIN----------------------
    # ::::::::::::::::::::::::::::::::

    num_steps = len(train_loader)
    start = time.time()
    total_losses = []
    # Divide the input 32x32 images into num_patches patch_sizexpatch_size patchs
    num_patches = (32//args.patch_size)**2

    for epoch in range(args.num_epochs):

        running_loss = 0.0
        current_losses = []
        for i, data in enumerate(train_loader, 0):

            # Get the images
            imgs = data[0]

            # Transform into patches
            patches = to_patches(imgs, args.patch_size)

            for patch in patches:
                # Transform the tensor into Variable
                v_patch = Variable(patch)
                target_tensor = Variable(torch.zeros(v_patch.size()), requires_grad=False)
                losses = []
                # Set gradients to Zero
                optimizer.zero_grad()
                for p in range(args.num_passes):
                    # Forward + Backward + Optimize
                    reconstructed_patches = model(v_patch, p)
                    losses.append(criterion(reconstructed_patches, target_tensor))

                    v_patch = reconstructed_patches
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

            # STATISTICS:

            if (i+1) % args.log_step == 0:
                print('(%s) [%d, %5d] loss: %.3f' %
                      (timeSince(start, ((epoch * num_steps + i + 1.0) / (args.num_epochs * num_steps))),
                       epoch + 1, i + 1, running_loss / args.log_step / num_patches))
                current_losses.append(running_loss/args.log_step/num_patches)
                running_loss = 0.0

            # SAVE:
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.model_path, args.model+'-p%d_b%d-%d_%d.pkl' %
                                        (args.patch_size, args.coded_size, epoch + 1, i + 1)))

        total_losses.append(current_losses)
        torch.save(model.state_dict(),
                   os.path.join(args.model_path,
                                +args.model + '-p%d_b%d-%d_%d.pkl' % (args.patch_size, args.coded_size, epoch + 1, i + 1)))

    print('__TRAINING DONE=================================================')


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
            patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.contiguous())
    return patches


#=============================================================================
# - PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='fc',
                        help='name of the model to be used: fc, fc_rec, conv, conv_rec, lstm ')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size')
    parser.add_argument('--coded_size', type=int, default=4,
                        help='number of bits representing the encoded patch')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='size for the encoded subdivision of the input image')
    parser.add_argument('--num_passes', type=int, default=16,
                        help='number of passes for recursive architectures')

    # ==================================================================================================================
    # OPTIMIZATION
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of iterations where the system sees all the data')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================================================================================================================
    # SAVING & PRINTING
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help='path were the models should be saved')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing the log info')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='step size for saving the trained models')


    #__________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)
