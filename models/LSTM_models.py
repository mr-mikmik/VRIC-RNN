import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from binary_layers import BinaryLayer

class LSTMEncoder(nn.Module):

    def __init__(self, coded_size, patch_size, batch_size):
        super(LSTMEncoder, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size
        self.batch_size = batch_size

        self.input_size = 512
        self.hidden_size = 512

        self.fc = nn.Linear(3*patch_size*patch_size, self.input_size)
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.binaryFC = nn.Linear(self.hidden_size, self.coded_size)
        self.binary = BinaryLayer()

    def init_state(self):
        h_1_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        c_1_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        h_2_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        c_2_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        return (h_1_0, c_1_0), (h_2_0, c_2_0)

    def forward(self, x, state):
        x = x.view(-1, 3*self.patch_size*self.patch_size)
        x = F.tanh(self.fc(x))
        h_out1, c_out1 = self.lstm1(x, state[0])
        h_out2, c_out2 = self.lstm2(h_out1, state[1])
        bits = self.binary(F.tanh(self.binaryFC(h_out2)))
        state = ((h_out1, c_out1), (h_out2, c_out2))
        return bits, state

class LSTMDecoder(nn.Module):

    def __init__(self, coded_size, patch_size, batch_size):
        super(LSTMDecoder, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size
        self.batch_size = batch_size

        self.input_size = 512
        self.hidden_size = 512

        self.fc1 = nn.Linear(self.coded_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 3*patch_size*patch_size)
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.input_size, self.hidden_size)

    def init_state(self):
        h_1_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        c_1_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        h_2_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        c_2_0 = Variable(torch.randn(self.batch_size, self.hidden_size))
        return (h_1_0, c_1_0), (h_2_0, c_2_0)

    def forward(self, x, state):
        x = F.tanh(self.fc1(x))
        h_out1, c_out1 = self.lstm1(x, state[0])
        h_out2, c_out2 = self.lstm2(h_out1, state[1])
        out = F.tanh(self.fc2(h_out2))
        state = ((h_out1, c_out1), (h_out2, c_out2))
        out = out.view(-1, 3, self.patch_size, self.patch_size)
        return out, state



class LSTMCore(nn.Module):

    def __init__(self, coded_size=4, patch_size=8, batch_size=4, num_passes=16):
        super(LSTMCore, self).__init__()
        self.num_passes = num_passes

        self.lstm_encoder = LSTMEncoder(coded_size, patch_size, batch_size)
        self.lstm_decoder = LSTMDecoder(coded_size, patch_size, batch_size)

    def forward(self, input_patch):
        encoder_state = self.lstm_encoder.init_state()
        decoder_state = self.lstm_decoder.init_state()

        patches = []
        bits = []

        for _ in range(self.num_passes):
            out_bits, encoder_state = self.lstm_encoder(input_patch, encoder_state)
            output_patch, decoder_state = self.lstm_decoder(out_bits, decoder_state)

            patches.append(output_patch)
            bits.append(bits)

            input_patch = input_patch-output_patch # Create the residual patch that will be the next input

        reconstructed_patch = sum(patches)

        return reconstructed_patch
