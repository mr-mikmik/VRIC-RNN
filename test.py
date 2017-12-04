import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import models

a = torch.rand(4,3,5,5)*2 -1
t = torch.rand(4,3,5,5)
v = Variable(a)
t = Variable(t)

k = v.view(-1,5)

b = models.BinaryLayer()
b2 = models.Binary2()
l = nn.Linear(5, 5)

class mm(nn.Module):
    def __init__(self):
        super(mm, self).__init__()
        self.lin = nn.Linear(5,5)
        self.bin = models.Binary2()

    def forward(self,x):
        x = x.view(-1, 5)
        x = self.lin(x)
        x = x.view(4,3,5,5)
        x = self.bin(x)
        return x

mmm = mm()
criterion = nn.MSELoss()

out1 = mmm(v)

out2 = l(k)
out2 = out2.view(4,3,5,5)
print out1
print out2

loss1 = criterion(out1, t)
loss2 = criterion(out2, t)
print 'Losses defined'
loss2.backward()
print 'Loss2 Backwarded'
loss1.backward()
print 'Loss1 Backwarded'
