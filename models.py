import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class BinaryLayer(Function):
    def forward(self, x):
        probs_tensor = torch.rand(x.size())
        probs_threshold = (x+1)/2
        x[probs_tensor<=probs_threshold] = 1-x[probs_tensor<=probs_threshold]
        x[probs_tensor>probs_threshold] = -x[probs_tensor>probs_threshold]-1
        return x


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
        self.w_bin = nn.Linear(512, self.coded_size)
        self.binary = BinaryLayer()

    def forward(self, x):
        """

        :param x: image typically a 8x8@3 patch image
        :return:
        """
        # Flatten the input
        x = x.view(-1, 3*self.patch_size**2)
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
        super(DecoderFC, self).__init__()
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
        x = x.view(-1, 3, self.patch_size, self.patch_size)
        return x


class ConvolutionalEncoder(nn.Module):

    def __init__(self, patch_size):
        super(ConvolutionalEncoder, self).__init__()
        self.patch_size = patch_size

        self.conv_1 = nn.Conv2d(3, 64, 4, stride=2)
        self.conv_2 = nn.Conv2d(64, 256, 3, stride=1)
        self.conv_3 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_1 = nn.Linear(512*6*6, 32)
        self.fc_2 = nn.Linear(32, 2)
        self.binary = BinaryLayer()

    def forward(self,x):
        x = F.tanh(self.conv_1(x))  # 32x32@3 --> 15x15@64
        x = F.tanh(self.conv_2(x))  # 15x15@64 --> 13x13@256
        x = F.tanh(self.conv_3(x))  # 13x13@256 --> 6x6@512
        x = x.view(-1, 512*6*6)
        x = F.tanh(self.fc_1(x))    # (6*6*512) --> 32
        x = F.tanh(self.fc_2(x))    # 32 --> 2
        x = self.binary(x)
        return x


class ConvolutionalDecoder(nn.Module):

    def __init__(self, patch_size):
        super(ConvolutionalDecoder, self).__init__()
        self.patch_size = patch_size
        # TODO: Modify and check the dimentions
        self.deconv_1 = nn.ConvTranspose2d(1,128,3,stride=2)
        self.deconv_2 = nn.ConvTranspose2d(128,512,2,stride=2)
        self.deconv_3 = nn.ConvTranspose2d(512,512,2,stride=2)


    def forward(self,x):
        x = F.tanh(self.deconv_1(x))
        x = F.tanh(self.deconv_2(x))
        x = F.tanh(self.deconv_3(x))
