from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def conv(in_channel, out_channel, kernel_size, stride, pad, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation))
def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))


class FlowWarp(nn.Module):
    def __init__(self):
        super(FlowWarp, self).__init__()
        self.h = -1
        self.w = -1

    def forward(self, x, f):
        # First, generate absolute coordinate from relative coordinates
        # f: N (rx,ry) oH oW
        # target: N oH oW (ax(width),ay(height))

        # Generate offset map
        width = x.size()[3]
        height = x.size()[2]
        if width != self.w or height != self.h:
            width_map = torch.arange(0, width, step=1).expand([height, width])
            height_map = torch.arange(0, height, step=1).unsqueeze(1).expand([height, width])
            self.offset_map = Variable(torch.stack([width_map, height_map], 2).cuda())
            self.w = width
            self.h = height
            self.scaler = Variable(1. / torch.cuda.FloatTensor([(self.w - 1) / 2, (self.h - 1) / 2]))

        f = f.permute(0, 2, 3, 1)  # N H W C
        f = f + self.offset_map  # add with dimension expansion
        f = f * self.scaler - 1  # scale to [-1,1]

        return F.grid_sample(x, f, mode='bilinear')  # eltwise multiply with broadcast

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
            self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x

class Concat(nn.Module):
    def __init__(self, axis = 0):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, inputs):
        return torch.cat(inputs, self.axis)

class AbsVal(nn.Module):
    def __init__(self):
        super(AbsVal,self).__init__()
    def forward(self, *input):
        return torch.abs(input)
