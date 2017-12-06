import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class BinaryLayer(nn.Module):
    def forward(self, x):
        probs_tensor = torch.rand(x.size())
        errors = Variable(torch.FloatTensor(x.size()))
        probs_threshold = torch.div(torch.add(x, 1), 2)
        alpha = 1-x[probs_tensor <= probs_threshold.data]
        beta = -x[probs_tensor > probs_threshold.data] - 1
        errors[probs_tensor <= probs_threshold.data] = alpha
        errors[probs_tensor > probs_threshold.data] = beta
        y = x + errors
        return y

    def backward(self, grad_output):
        return grad_output


class Binary2(Function):
    def forward(self, x):
        return torch.sign(x)

    def backward(self, grad_output):
        #input_t = self.saved_tensors
        #grad_output[input_t>1] = 0
        #grad_output[input_t<-1] = 0
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
        self.w_bin = nn.Linear(512, self.coded_size)
        self.binary = Binary2()

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

    def __init__(self, coded_size, patch_size=32):
        super(ConvolutionalEncoder, self).__init__()
        self.patch_size = patch_size

        self.conv_1 = nn.Conv2d(3, 64, 4, stride=2)
        self.conv_2 = nn.Conv2d(64, 256, 3, stride=1)
        self.conv_3 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_1 = nn.Linear(512*6*6, coded_size*8)
        self.fc_2 = nn.Linear(coded_size*8, coded_size)
        self.binary = Binary2()

    def forward(self,x):
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
        self.deconv_1 = nn.ConvTranspose2d(512,128,3,stride=2)
        self.deconv_2 = nn.ConvTranspose2d(128,64,3,stride=2)
        self.deconv_3 = nn.ConvTranspose2d(64,8,3,stride=1)
        self.deconv_4 = nn.ConvTranspose2d(8,3,4,stride=1)


    def forward(self,x):
        x = F.tanh(self.fc_1(x))
        x = F.tanh(self.fc_2(x))
        x = x.view(-1,512,6,6)
        x = F.tanh(self.deconv_1(x))
        x = F.tanh(self.deconv_2(x))
        x = F.tanh(self.deconv_3(x))
        x = F.tanh(self.deconv_4(x))
        return x
