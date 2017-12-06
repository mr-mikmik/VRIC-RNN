import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from binary_layers import BinaryLayer

class ConvolutionalEncoder(nn.Module):

    def __init__(self, coded_size, patch_size=32):
        super(ConvolutionalEncoder, self).__init__()
        self.patch_size = patch_size

        self.conv_1 = nn.Conv2d(3, 64, 4, stride=2)
        self.conv_2 = nn.Conv2d(64, 256, 3, stride=1)
        self.conv_3 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_1 = nn.Linear(512*6*6, coded_size*8)
        self.fc_2 = nn.Linear(coded_size*8, coded_size)
        self.binary = BinaryLayer()

    def forward(self, x):
        x = F.tanh(self.conv_1(x))  # 32x32@3 --> 15x15@64
        x = F.tanh(self.conv_2(x))  # 15x15@64 --> 13x13@256
        x = F.tanh(self.conv_3(x))  # 13x13@256 --> 6x6@512
        x = x.view(-1, 512*6*6)
        x = F.tanh(self.fc_1(x))    # (6*6*512) --> coded_size*8
        x = F.tanh(self.fc_2(x))    # coded_size*8 --> coded_size
        x = self.binary(x)
        return x


class ConvolutionalDecoder(nn.Module):

    def __init__(self, coded_size, patch_size=32):
        super(ConvolutionalDecoder, self).__init__()
        self.patch_size = patch_size
        self.fc_1 = nn.Linear(coded_size, coded_size*8)
        self.fc_2 = nn.Linear(coded_size*8, 512*6*6)
        # TODO: Modify and check the dimentions
        self.deconv_1 = nn.ConvTranspose2d(512, 128, 3, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.deconv_3 = nn.ConvTranspose2d(64, 8, 3, stride=1)
        self.deconv_4 = nn.ConvTranspose2d(8, 3, 4, stride=1)


    def forward(self,x):
        x = F.tanh(self.fc_1(x))
        x = F.tanh(self.fc_2(x))
        x = x.view(-1, 512, 6, 6)
        x = F.tanh(self.deconv_1(x))
        x = F.tanh(self.deconv_2(x))
        x = F.tanh(self.deconv_3(x))
        return x


class ConvolutionalCore(nn.Module):

    def __init__(self, coded_size=64, patch_size=32):
        super(ConvolutionalCore, self).__init__()

        self.conv_encoder = ConvolutionalEncoder(coded_size, patch_size)
        self.conv_decoder = ConvolutionalDecoder(coded_size, patch_size)

    def forward(self, input_patch):
        out_bits = self.conv_encoder(input_patch)
        output_patch = self.conv_decoder(out_bits)

        return output_patch
