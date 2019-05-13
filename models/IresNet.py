from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functiongal as F
import math
import numpy as np

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel))

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size = kernel_size,
            padding=pad,
            stride = stride),
        nn.BatchNorm3d(out_channel))

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        # out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out = x + out
        return out
