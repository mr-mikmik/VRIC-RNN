import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from binary_layers import BinaryLayer


class EncoderFC(nn.Module):
    """
    FC Encoder composed by 3 512-units fully-connected layers
    """
    def __init__(self, coded_size, patch_size):
        super(EncoderFC, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size

        self.fc1 = nn.Linear(3 * patch_size * patch_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.w_bin = nn.Linear(512, self.coded_size)
        self.binary = BinaryLayer()

    def forward(self, x):
        """
        :param x: image typically a 8x8@3 patch image
        :return:
        """
        # Flatten the input
        x = x.view(-1, 3 * self.patch_size ** 2)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.w_bin(x))
        x = self.binary(x)
        return x


class DecoderFC(nn.Module):
    """
    FC Decoder composed by 3 512-units fully-connected layers
    """
    def __init__(self, coded_size, patch_size):
        super(DecoderFC, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size

        self.fc1 = nn.Linear(coded_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.last_layer = nn.Linear(512, patch_size * patch_size * 3)

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


class CoreFC(nn.Module):

    def __init__(self, coded_size, patch_size):
        super(CoreFC, self).__init__()

        self.fc_encoder = EncoderFC(coded_size, patch_size)
        self.fc_decoder = DecoderFC(coded_size, patch_size)

    def forward(self, x):
        bits = self.fc_encoder(x)
        out = self.fc_decoder(bits)
        return out

class ResidualCoreFC(nn.Module):

    def __init__(self, coded_size=4, patch_size=8, num_passes=16):
        super(ResidualCoreFC, self).__init__()
        self.num_passes = num_passes

        self.encoders = []
        self.decoders = []

        for _ in range(self.num_passes):

            self.encoders.append(EncoderFC(coded_size, patch_size))
            self.decoders.append(DecoderFC(coded_size, patch_size))

    def forward(self, input_patch):
        patches = []
        bits = []

        for i in range(self.num_passes):
            out_bits = self.encoders[i](input_patch)
            output_patch = self.decoders[i](out_bits)

            patches.append(output_patch)
            bits.append(out_bits)

            input_patch = input_patch.clone() - output_patch

        reconstructed_patch = sum(patches)

        return reconstructed_patch


class Residual2CoreFC(nn.Module):

    def __init__(self, coded_size=4, patch_size=8, num_passes=16):
        super(Residual2CoreFC, self).__init__()
        self.num_passes = num_passes

        self.encoders = nn.ModuleList([EncoderFC(coded_size, patch_size) for i in range(num_passes)])
        self.decoders = nn.ModuleList([DecoderFC(coded_size, patch_size) for i in range(num_passes)])

    def forward(self, input_patch, pass_num):

        out_bits = self.encoders[pass_num](input_patch)
        output_patch = self.decoders[pass_num](out_bits)

        residual_patch = input_patch - output_patch
        return residual_patch


class ResidualCoreFC3(nn.Module):

    def __init__(self, coded_size=4, patch_size=8, num_passes=16):
        super(ResidualCoreFC3, self).__init__()
        self.num_passes = num_passes

        self.encoders = nn.ModuleList([EncoderFC(coded_size, patch_size) for i in range(num_passes)])
        self.encoders = nn.ModuleList([EncoderFC(coded_size, patch_size) for i in range(num_passes)])

    def forward(self, input_patch, pass_num):


        out_bits = self.encoders[pass_num](input_patch)
        output_patch = self.decoders[pass_num](out_bits)

        residual_patch = input_patch - output_patch
        return residual_patch
