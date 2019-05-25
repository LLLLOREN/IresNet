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
            dilation=dilation))


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

class basisiResNet(nn.Module):
    def __init__(self):
        super(basisiResNet, self).__init__()
        #features extraction
        self.conv1 = nn.Sequential(convbn(3, 64, 7, 2, 3, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(convbn(64, 128, 5, 2, 2, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        #for corr
        self.corr_conv3 = nn.Sequential(convbn(128, 256, 3, 2, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.corr_conv3_1 = nn.Sequential(convbn(256, 256, 3, 1, 1, 1),
                                          nn.LeakyReLU(negative_slope=0.1))
        self.corr_deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.corr_fusion2 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                          nn.LeakyReLU(negative_slope=0.1))
        self.conv_redir = nn.Sequential(convbn(128, 64, 1, 1, 0, 1),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.conv3 = nn.Sequential(convbn(145, 256, 5, 2, 2, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv3_1 = nn.Sequential(convbn(256, 256, 3, 1, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv4 = nn.Sequential(convbn(256, 512, 3, 2,  1, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv4_1 = nn.Sequential(convbn(512, 512, 3, 1, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv5 = nn.Sequential(convbn(512, 512, 3, 2, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv5_1 = nn.Sequential(convbn(512, 512, 3, 1, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv6 = nn.Sequential(convbn(512, 1024, 3, 2, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv6_1 = nn.Sequential(convbn(1024, 1024, 3, 1, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))

        #disparity regression
        self.upsample_flow = nn.Sequential(nn.ConvTranspose2d(1, 1, 4, 2, 1))
        #(6-1) regression
        self.Convolution1 = nn.Sequential(convbn(1024, 1, 3, 1, 1, 1))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(5-1)
        self.Convolution2 = nn.Sequential(convbn(1025, 512, 3, 1, 1, 1))
        self.Convolution3 = nn.Sequential(convbn(512, 1, 3, 1, 1, 1))
        #(5-2)
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(4-1)
        self.Convolution4 = nn.Sequential(convbn(769, 256, 3, 1, 1, 1))
        self.Convolution5 = nn.Sequential(convbn(256, 1, 3, 1, 1, 1))
        #(4-2)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(3-1)
        self.Convolution6 = nn.Sequential(convbn(385, 128, 3, 1, 1, 1))
        self.Convolution7 = nn.Sequential(convbn(128, 1, 3, 1, 1, 1))
        #(3-2)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(2-1)
        self.Convolution8 = nn.Sequential(convbn(193, 64, 3, 1, 1, 1))
        self.Convolution9 = nn.Sequential(convbn(64, 1, 3, 1, 1, 1))
        #(2-2)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(1-1)
        self.Convolution10 = nn.Sequential(convbn(97, 32, 3, 1, 1, 1))
        self.Convolution11 = nn.Sequential(convbn(32, 1, 3, 1, 1, 1))

        #Multi-Scale-Full-Disparity
        # skip connections
        self.up_conv1ab = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                      nn.LeakyReLU(negative_slope=0.1))
        self.up_conv2ab = nn.Sequential(nn.ConvTranspose2d(128, 32, 8, 4, 2),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.up_conv12 = nn.Sequential(convbn(64, 32, 1, 1, 0, 1),
                                       nn.LeakyReLU(negative_slope=0.1))

        # main branch
        self.deconv0 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))

        # predict
        self.Convolution21 = nn.Sequential(convbn(65, 32, 3, 1, 1, 1))
        self.Convolution22 = nn.Sequential(convbn(32, 1, 3, 1, 1, 1))

        self.subupsample_felow6 = nn.Sequential(nn.ConvTranspose2d(1, 1, 128, 64, 32),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.subupsample_felow5 = nn.Sequential(nn.ConvTranspose2d(1, 1, 64, 32, 16),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.subupsample_felow4 = nn.Sequential(nn.ConvTranspose2d(1, 1, 32, 16, 8),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.subupsample_felow3 = nn.Sequential(nn.ConvTranspose2d(1, 1, 16, 8, 4),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.subupsample_felow2 = nn.Sequential(nn.ConvTranspose2d(1, 1, 8, 4, 2),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.subupsample_felow1 = nn.Sequential(nn.ConvTranspose2d(1, 1, 4, 2, 1),
                                                nn.LeakyReLU(negative_slope=0.1))
        self.Convolution_predict_from_multi_res = nn.Sequential(convbn(7, 1, 1, 1, 0, 1),
                                                                nn.LeakyReLU(negative_slope=0, inplace=True))

    def forward(self, Limg, Rimg):

        #feature Extraction
        conv1a = self.conv1(Limg)
        conv1b = self.conv1(Rimg)
        conv2a = self.conv2(conv1a)
        conv2b = self.conv2(conv1b)
        #for corr
        corr_conv3a = self.corr_conv3(conv2a)
        corr_conv3b = self.corr_conv3(conv2b)
        corr_conv3_1a = self.corr_conv3_1(corr_conv3a)
        corr_conv3_1b = self.corr_conv3_1(corr_conv3b)
        corr_deconv3a = self.corr_deconv3(corr_conv3_1a)
        corr_deconv3b = self.corr_deconv3(corr_conv3_1b)
        corr_deconv3a = self.ReLU(corr_deconv3a)
        corr_deconv3a = self.ReLU(corr_deconv3b)
        concat_edcorr_2a = torch.cat((corr_deconv3a, conv2a))
        concat_edcorr_2b = torch.cat((corr_deconv3b, conv2b))
        edcorr_2a = self.corr_fusion2(concat_edcorr_2a)
        edcorr_2b = self.corr_fusion2(concat_edcorr_2b)

        corr = correlation(edcorr_2a, edcorr_2b, 40)
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
        #features extraction finished

        #disparity regression
        #(6-1) regression 12*6 --loss for disparity
        predict_flow = self.Convolution1(conv6_1)
        subupsampled_flow6 = self.subupsample_felow6(predict_flow)
        deconv = self.deconv5(conv6_1)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat([conv5_1, deconv, upsampled_flow])
        #(5-1) 24*12 --loss for disparity
        concat = self.Convolution2(output)
        predict_flow5 = self.Convolution3(concat)
        subupsampled_flow5 = self.subupsample_felow5(predict_flow)
        #(5-2) 24*12 --prepare features for next stage
        deconv = self.deconv4(concat)
        upsampled_flow = self.upsample_flow(predict_flow5)
        output = torch.cat([conv4_1, deconv, upsampled_flow])
        #(4-1) 48*24 --loss for disparity
        concat = self.Convolution4(output)
        predict_flow = self.Convolution5(concat)
        subupsampled_flow4 = self.subupsample_felow4(predict_flow)
        #(4-2) 48*24 --prepare features for next stage
        deconv = self.deconv3(concat)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat([conv3_1, deconv, upsampled_flow])
        #(3-1) regression 96*48 --loss for diparity
        concat = self.Convolution6(output)
        predict_flow = self.Convolution7(concat)
        subupsampled_flow3 = self.subupsample_felow3(predict_flow)
        #(3-2) regression 96*48 --prepare features for next stage
        deconv = self.deconv2(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat([conv2a, deconv, upsampled_flow])
        # (2-1) regression 192*96 -- loss for disparity
        concat = self.Convolution8(output)
        predict_flow = self.Convolution9(concat)
        subupsampled_flow2 = self.subupsample_felow2(predict_flow)
        # (2-2) regression 192*96  -- prepare features for next stage
        deconv = self.deconv1(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = torch.cat([conv2a, deconv, upsampled_flow])
        # (1-1) regression 384*192 -- loss for disparity
        concat = self.Convolution10(output)
        predict_flow1 = self.Convolution11(output)
        subupsampled_flow1 = self.subupsample_felow1(predict_flow)

        #Multi-Scale-Full-Disparity
        up_conv1a = self.up_conv1ab(conv1a)
        up_conv1b = self.up_conv1ab(conv1b)
        up_conv2a = self.up_conv2ab(conv2a)
        up_conv2b = self.up_conv2ab(conv2b)
        concat_up_conv1a2a = torch.cat([up_conv1a, up_conv2a], 1)
        concat_up_conv1b2b = torch.cat([up_conv1b, up_conv2b], 1)
        up_conv1a2a = self.up_conv12(concat_up_conv1a2a)
        up_conv1b2b = self.up_conv12(concat_up_conv1b2b)

        #upsample disparity
        upsampled_flow = self.upsample_flow(predict_flow1)

        #main branch
        deconv = self.deconv0(concat)
        output = torch.cat((up_conv1a2a, deconv, upsampled_flow))

        # predict
        concat = self.Convolution21(output)
        predict_flow0 = self.Convolution22(concat)

        # concat multi-res prediction
        concat_a = torch.cat([subupsampled_flow6,
                              subupsampled_flow5,
                              subupsampled_flow4,
                              subupsampled_flow3,
                              subupsampled_flow2,
                              subupsampled_flow1,
                              predict_flow0])
        final_prediction = self.Convolution_predict_from_multi_res(concat_a)

        return final_prediction,conv1a, conv1b, up_conv1b2b

class refinementSub(nn.Module):
    def __init__(self, conv1a, conv1b, final_prediction):
        super(refinementSub,self).__init__()
        self.conv1a = conv1a
        self.conv1b = conv1b
        self.input = final_prediction
