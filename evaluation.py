import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse
import scipy.misc
import numpy as np

from PIL import Image

import models


def main(args):
    # Create the model directory if does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Normalize the input images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    dataiter = iter(testloader)

    print('Loading Models')
    # Initialize the models
    model = models.setup(args)

    # Load the SAVED model
    path_to_model = os.path.join(args.model_path, +args.model+'-p%d_b%d-%d_%d.pkl' %
                                 (args.patch_size, args.coded_size, args.load_iter, args.load_iter))
    model.load_state_dict(torch.load(path_to_model))

    print('Starting eval:::::::::::::::::')
    for i in range(5):
        imgs, _ = dataiter.next()
        imsave(torchvision.utils.make_grid(imgs), 'prova_'+str(i))

        # Patch the image:
        patches = to_patches(imgs, args.batch_size)
        r_patches = []  # Reconstructed Patches
        for p in patches:
            if args.residual:
                outputs = model.sample(Variable(p))
            else:
                outputs = model(Variable(p))
            r_patches.append(outputs)
        # Transform the patches into the image
        outputs = reconstruct_patches(r_patches)
        imsave(torchvision.utils.make_grid(outputs), 'prova_'+str(i)+'_decoded')


#==============================================
# - CUSTOM FUNCTIONS
#==============================================

def imsave(img, name):
    img = img / 2 + 0.5     # unnormalize
    saving_path = os.path.join(args.output_path, name+'.png')
    torchvision.utils.save_image(img, saving_path)


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


#=============================================================================
# - PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='fc',
                        help='name of the model to be used: fc, conv, lstm ')
    parser.add_argument('--residual', type=bool, default=False,
                        help='Set True if the model is residual, otherwise False')
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
    parser.add_argument('--output_path', type=str, default='./test_imgs/')

    parser.add_argument('--load_iter', type=int, default=12500,
                        help='iteration which the model to be loaded was saved')
    parser.add_argument('--load_epoch', type=int, default=3,
                        help='epoch in which the model to be loaded was saved')

    # __________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)
