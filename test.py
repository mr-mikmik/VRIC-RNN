import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import models

conv_1 = nn.Conv2d(3, 64, 4, stride=2)
conv_2 = nn.Conv2d(64, 256, 3, stride=1)
conv_3 = nn.Conv2d(256, 512, 3, stride=2)
conv_x = nn.Conv2d(3, 64, 3, stride=2)

input_t = Variable(torch.randn(4,3,8,8))



c_1 = nn.Conv2d(3, 64, 2, stride=1)
c_2 = nn.Conv2d(64, 256, 3, stride=2)
c_3 = nn.Conv2d(256,512,3, stride=2)

d_1 = nn.ConvTranspose2d(512,128,3,stride=2)
d_2 = nn.ConvTranspose2d(128,64,2,stride=2)
d_3 = nn.ConvTranspose2d(64,3,2,stride=1)
d_4 = nn.ConvTranspose2d(32,3,2,stride=1)

o_1 = c_1(input_t)
o_2 = c_2(o_1)
o_3 = c_3(o_2)

k = Variable(torch.randn(4,512,1,1))
r_1 = d_1(k)
r_2 = d_2(r_1)
r_3 = d_3(r_2)