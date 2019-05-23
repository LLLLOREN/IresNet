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

class iresNet(nn.Module):
    def __init__(self):
        super(iresNet, self).__init__()
        self.conv = nn.Sequential()

        #features extraction
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

        #disparity regression
        self.add_module("upsample_flow", nn.ConvTranspose2d(1, 1, 4, 2, 1))
        #(6-1) regression
        self.add_module("Convolution1", convbn(1024, 1, 3, 1, 1, 1))
        self.add_module("deconv5", nn.ConvTranspose2d(1024, 512, 4, 2, 1))
        #(5-1)
        self.add_module("Convolution2",convbn(1025, 512, 3, 1, 1, 1))
        self.add_module("Convolution3", convbn(512, 1, 3, 1, 1, 1))
        #(5-2)
        self.add_module("deconv4", nn.ConvTranspose2d(512, 256, 4, 2, 1))
        #(4-1)
        self.add_module("Convolution4", convbn(769, 256, 3, 1, 1, 1))
        self.add_module("Convolution5", convbn(256, 1, 3, 1, 1, 1))
        #(4-2)
        self.add_module("deconv3", nn.ConvTranspose2d(256, 128, 4, 2, 1))
        #(3-1)
        self.add_module("Convolution6", convbn(385, 128, 3, 1, 1, 1))
        self.add_module("Convolution7", convbn(128, 1, 3, 1, 1, 1))
        #(3-2)
        self.add_module("deconv2", nn.ConvTranspose2d(128, 64, 4, 2, 1))
        #(2-1)
        self.add_module("Convolution8", convbn(193, 64, 3, 1, 1, 1))
        self.add_module("Convolution9",convbn(64, 1, 3, 1, 1, 1))
        #(2-2)
        self.add_module("deconv1", nn.ConvTranspose2d(64, 32, 4, 2, 1))
        #(1-1)
        self.add_module("Convolution10", convbn(97, 32, 3, 1, 1, 1))
        self.add_module("Convolution11", convbn(32, 1, 3, 1, 1, 1))

        #Multi-Scale-Full-Disparity
        # skip connections
        self.add_module("upconv11", nn.ConvTranspose2d(64, 32, 4, 2, 1))
        self.add_module("upconv21", nn.ConvTranspose2d(128, 32, 8, 4, 2))
        self.add_module("upconv12", convbn_relu(64, 32, 1, 1, 0, 1))
        # upsample disparity


    def forward(self, Limg, Rimg):

        #feature Extraction
        conv1a = self.conv1(Limg)
        conv1b = self.conv1(Rimg)
        conv2a = self.conv2(conv1a)
        conv2b = self.conv2(conv1b)
        corr = correlation(conv2a, conv2b, 40)
        conv_redir = self.conv_redir(conv2a)
        output = torch.cat((corr, conv_redir),1)
        output = self.conv3(output)
        conv3_1= self.conv3_1(output)
        output = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(output)
        output = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(output)
        output = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(output)

        #disparity regression
        #(6-1) regression 12*6 --loss for disparity
        predict_flow = self.Convolution1(conv6_1)
        deconv = self.deconv5(conv6_1)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat((conv5_1, deconv, upsampled_flow))
        #(5-1) 24*12 --loss for disparity
        concat = self.Convolution2(output)
        predict_flow = self.Convolution3(concat)
        #(5-2) 24*12 --prepare features for next stage
        deconv = self.deconv4(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat((conv4_1, deconv, upsampled_flow))
        #(4-1) 48*24 --loss for disparity
        concat = self.Convolution4(output)
        predict_flow = self.Convolution5(concat)
        #(4-2) 48*24 --prepare features for next stage
        deconv = self.deconv3(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat((conv3_1, deconv, upsampled_flow))
        #(3-1) regression 96*48 --loss for diparity
        concat = self.Convolution6(output)
        predict_flow = self.Convolution7(concat)
        #(3-2) regression 96*48 --prepare features for next stage
        deconv = self.deconv2(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat((conv2a, deconv, upsampled_flow))
        # (2-1) regression 192*96 -- loss for disparity
        concat = self.Convolution8(output)
        predict_flow = self.Convolution9(concat)
        # (2-2) regression 192*96 -- prepare features for next stage
        deconv = self.deconv1(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat((conv2a, deconv, upsampled_flow))
        # (1-1) regression 384*192 -- loss for disparity
        concat = self.Convolution10(output)
        predict_flow = self.Convolution11(output)

        #Multi-Scale-Full-Disparity
        up_conv1a = self.upconv11(conv1a)
        up_conv1b = self.upconv11(conv1b)
        up_conv1a = self.ReLU(up_conv1a)
        up_conv1b = self.ReLu(up_conv1b)
        return output
