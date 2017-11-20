import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class BinaryLayer(Function):
    def forward(self, input):
        return torch.sign(input)

    def backward(self, grad_output):
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class EncoderFC(nn.Module):
    """
    FC Encoder composed by 3 512-units fully-connected layers
    """
    def __init__(self, coded_size, patch_size):
        super(EncoderFC, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size 
        
        self.fc1 = nn.Linear(3*patch_size*patch_size,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.w_bin = nn.linear(512, self.coded_size)
        self.binary = BinaryLayer()

    def forward(self, x):
        """

        :param x: image typically a 8x8@3 patch image
        :return:
        """
        # Flatten the input
        x = x.view(-1,3*self.patch_size**2)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.w_bin(x))
        x = self.binary(x)
        return x


class DecoderFC(nn.Module):
    """
        FC Encoder composed by 3 512-units fully-connected layers
        """

    def __init__(self, coded_size, patch_size):
        super(EncoderFC, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size

        self.fc1 = nn.Linear(coded_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.last_layer = nn.Linear(512, patch_size*patch_size*3)


    def forward(self, x):
        """

        :param x: encoded features
        :return:
        """
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.last_layer(x)
        x = x.view(-1, self.patch_size, self.patch_size)
        return x


class ConvolutionalEncoder(nn.Module):

    def __init__(self, patch_size):
        super(ConvolutionalEncoder, self).__init__()
        self.patch_size = patch_size

        conv_1 = nn.Conv2d(3, 64, 3, stride=2)
        conv_2 = nn.Conv2d(64, 256, 3, stride=1)
        conv_3 = nn.Conv2d(256, 512, 3, stride=2)
        self.binary = BinaryLayer()

    def forward(self,x):
