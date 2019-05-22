from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
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

def convbn_relu(in_channel, out_channel, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(negative_slope = 0.1))


def correlation(x, y, max_disp):
    w = y.size()[2]
    w = torch.Tensor()
    corr_tensor = []
    for i in range(-max_disp, 0, 1):
        shifted = F.pad(y[:, :, 0:w + i, :],[[0, 0], [0, 0], [-i, 0],[0, 0]], mode = 'constant')
        corr = torch.mean(shifted.mul(w),dim = 3)
        corr_tensor.append(corr)
    for i in range(max_disp + 1):
        shifted = F.pad(x[:, :, i:, :],[[0, 0], [0, 0], [-i, 0], [0, 0]], mode = 'constant')
        corr = torch.mean(shifted.mul(y), dim = 3)
        corr_tensor.append(corr)
    corr_tensor = torch.stack(corr_tensor)
    return corr_tensor.permute(1, 2, 3, 0)

#
# class BasicBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride, downsample, pad, dilation):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             convbn(in_channel, out_channel, kernel_size, stride, pad, dilation),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True))
#         # self.conv2 = nn.Sequential(convbn(out_channel, out_channel, 3, 1, pad, dilation))
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         out = self.conv1(x)
#
#         # out = self.conv2(out)
#
#         if self.downsample is not None:
#             x = self.downsample(x)
#
#         out = x + out
#         return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.FloatTensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out

class featuresExtraction(nn.Module):
    def __init__(self):
        super(featuresExtraction, self).__init__()
        self.conv = torch.nn.Sequential()
        self.add_module("ReLU", nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.add_module("conv1",convbn_relu(3, 64, 7, 2, 3, 1))
        self.add_module("conv2", convbn_relu(64, 128, 5, 2, 2, 1))
        self.add_module("conv_redir", convbn_relu(128, 64, 1, 1, 0, 1))
        self.add_module("conv3", convbn_relu(145, 256, 5, 2, 2, 1))
        self.add_module("conv3_1", convbn_relu(256, 256, 3, 1, 1, 1))
        self.add_module("conv4", convbn_relu(256, 512, 3, 2,  1, 1))
        self.add_module("conv4_1", convbn_relu(512, 512, 3, 1, 1, 1))
        self.add_module("conv5", convbn_relu(512, 512, 3, 2, 1, 1))
        self.add_module("conv5_1", convbn_relu(512, 512, 3, 1, 1, 1))
        self.add_module("conv6", convbn_relu(512, 1024, 3, 2, 1, 1))
        self.add_module("conv6_1", convbn_relu(1024, 1024, 3, 1, 1, 1))
    def forward(self, Limg, Rimg):
        L = self.conv1(Limg)
        R = self.conv1(Rimg)
        L = self.conv2(L)
        R = self.conv2(R)
        corr = correlation(L, R, 40)
        conv_redir = self.conv_redir(L)
        output = torch.cat((corr, conv_redir),1)
        output = self.conv3(output)
        output = self.conv3_1(output)
        output = self.conv4(output)
        output = self.conv4_1(output)
        output = self.conv5(output)
        output = self.conv5_1(output)
        output = self.conv6(output)
        output = self.conv6_1(output)

        return output


