from __future__ import print_function
from .correlation_package.correlation import Correlation as corr
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from .layers import *


class basisiResNet(nn.Module):
    def __init__(self):
        super(basisiResNet, self).__init__()

        #features extraction
        self.conv1 = nn.Sequential(conv(3, 64, 7, 2, 3),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(conv(64, 128, 5, 2, 2),
                                   nn.LeakyReLU(negative_slope=0.1))
        #for corr
        self.corr_conv3 = nn.Sequential(conv(128, 256, 3, 2, 1),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.corr_conv3_1 = nn.Sequential(conv(256, 256, 3, 1, 1),
                                          nn.LeakyReLU(negative_slope=0.1))
        self.corr_deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.corr_fusion2 = nn.Sequential(conv(256, 128, 3, 1, 1),
                                          nn.LeakyReLU(negative_slope=0.1))
        self.corr_Concat3 = nn.Sequential(Concat())
        self.corr = nn.Sequential(corr(50, 1, 50, 1, 1))

        self.conv_redir = nn.Sequential(conv(128, 64, 1, 1, 0),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.concat2 = nn.Sequential(Concat(1))
        self.conv3 = nn.Sequential(conv(145, 256, 5, 2, 2),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv3_1 = nn.Sequential(conv(256, 256, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv4 = nn.Sequential(conv(256, 512, 3, 2,  1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv4_1 = nn.Sequential(conv(512, 512, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv5 = nn.Sequential(conv(512, 512, 3, 2, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv5_1 = nn.Sequential(conv(512, 512, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        self.conv6 = nn.Sequential(conv(512, 1024, 3, 2, 1),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.conv6_1 = nn.Sequential(conv(1024, 1024, 3, 1, 1),
                                     nn.LeakyReLU(negative_slope=0.1))

        #disparity regression
        self.upsample_flow = nn.Sequential(nn.ConvTranspose2d(1, 1, 4, 2, 1))
        self.concat = nn.Sequential(Concat())
        #(6-1) regression
        self.Convolution1 = nn.Sequential(conv(1024, 1, 3, 1, 1))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(5-1)
        self.Convolution2 = nn.Sequential(conv(1025, 512, 3, 1, 1))
        self.Convolution3 = nn.Sequential(conv(512, 1, 3, 1, 1))
        #(5-2)
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(4-1)
        self.Convolution4 = nn.Sequential(conv(769, 256, 3, 1, 1))
        self.Convolution5 = nn.Sequential(conv(256, 1, 3, 1, 1))
        #(4-2)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(3-1)
        self.Convolution6 = nn.Sequential(conv(385, 128, 3, 1, 1))
        self.Convolution7 = nn.Sequential(conv(128, 1, 3, 1, 1))
        #(3-2)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(2-1)
        self.Convolution8 = nn.Sequential(conv(193, 64, 3, 1, 1))
        self.Convolution9 = nn.Sequential(conv(64, 1, 3, 1, 1))
        #(2-2)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))
        #(1-1)
        self.Convolution10 = nn.Sequential(conv(97, 32, 3, 1, 1))
        self.Convolution11 = nn.Sequential(conv(32, 1, 3, 1, 1))

        #Multi-Scale-Full-Disparity
        # skip connections
        self.up_conv1ab = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                      nn.LeakyReLU(negative_slope=0.1))
        self.up_conv2ab = nn.Sequential(nn.ConvTranspose2d(128, 32, 8, 4, 2),
                                        nn.LeakyReLU(negative_slope=0.1))
        self.up_conv12 = nn.Sequential(conv(64, 32, 1, 1, 0),
                                       nn.LeakyReLU(negative_slope=0.1))

        # main branch
        self.deconv0 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1),
                                     nn.LeakyReLU(negative_slope=0.1))

        # predict
        self.Convolution21 = nn.Sequential(conv(65, 32, 3, 1, 1, 1))
        self.Convolution22 = nn.Sequential(conv(32, 1, 3, 1, 1, 1))

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
        self.Convolution_predict_from_multi_res = nn.Sequential(conv(7, 1, 1, 1, 0),
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
        concat_edcorr_2a = self.corr_Concat3([corr_deconv3a, conv2a])
        concat_edcorr_2b = self.corr_Concat3([corr_deconv3b, conv2b])
        edcorr_2a = self.corr_fusion2(concat_edcorr_2a)
        edcorr_2b = self.corr_fusion2(concat_edcorr_2b)

        corr = self.corr(edcorr_2a, edcorr_2b)
        conv_redir = self.conv_redir(conv2a)
        output = self.concat2([corr, conv_redir])
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
        output = self.concat([conv5_1, deconv, upsampled_flow])
        #(5-1) 24*12 --loss for disparity
        concat = self.Convolution2(output)
        predict_flow5 = self.Convolution3(concat)
        subupsampled_flow5 = self.subupsample_felow5(predict_flow)
        #(5-2) 24*12 --prepare features for next stage
        deconv = self.deconv4(concat)
        upsampled_flow = self.upsample_flow(predict_flow5)
        output = self.concat([conv4_1, deconv, upsampled_flow])
        #(4-1) 48*24 --loss for disparity
        concat = self.Convolution4(output)
        predict_flow = self.Convolution5(concat)
        subupsampled_flow4 = self.subupsample_felow4(predict_flow)
        #(4-2) 48*24 --prepare features for next stage
        deconv = self.deconv3(concat)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = self.concat([conv3_1, deconv, upsampled_flow])
        #(3-1) regression 96*48 --loss for diparity
        concat = self.Convolution6(output)
        predict_flow = self.Convolution7(concat)
        subupsampled_flow3 = self.subupsample_felow3(predict_flow)
        #(3-2) regression 96*48 --prepare features for next stage
        deconv = self.deconv2(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = self.concat([conv2a, deconv, upsampled_flow])
        # (2-1) regression 192*96 -- loss for disparity
        concat = self.Convolution8(output)
        predict_flow = self.Convolution9(concat)
        subupsampled_flow2 = self.subupsample_felow2(predict_flow)
        # (2-2) regression 192*96  -- prepare features for next stage
        deconv = self.deconv1(concat)
        deconv = self.ReLU(deconv)
        upsampled_flow = self.upsample_flow(predict_flow)
        output = self.concat2([conv2a, deconv, upsampled_flow])
        # (1-1) regression 384*192 -- loss for disparity
        concat = self.Convolution10(output)
        predict_flow1 = self.Convolution11(output)
        subupsampled_flow1 = self.subupsample_felow1(predict_flow)

        #Multi-Scale-Full-Disparity
        up_conv1a = self.up_conv1ab(conv1a)
        up_conv1b = self.up_conv1ab(conv1b)
        up_conv2a = self.up_conv2ab(conv2a)
        up_conv2b = self.up_conv2ab(conv2b)
        concat_up_conv1a2a = self.concat2([up_conv1a, up_conv2a])
        concat_up_conv1b2b = self.concat2([up_conv1b, up_conv2b])
        up_conv1a2a = self.up_conv12(concat_up_conv1a2a)
        up_conv1b2b = self.up_conv12(concat_up_conv1b2b)

        #upsample disparity
        upsampled_flow = self.upsample_flow(predict_flow1)

        #main branch
        deconv = self.deconv0(concat)
        output = self.concat([up_conv1a2a, deconv, upsampled_flow])

        # predict
        concat = self.Convolution21(output)
        predict_flow0 = self.Convolution22(concat)

        # concat multi-res prediction
        concat_a = self.concat([subupsampled_flow6,
                              subupsampled_flow5,
                              subupsampled_flow4,
                              subupsampled_flow3,
                              subupsampled_flow2,
                              subupsampled_flow1,
                              predict_flow0])
        final_prediction = self.Convolution_predict_from_multi_res(concat_a)

        return final_prediction,conv1a, conv1b, up_conv1b2b

class refinementSub(nn.Module):
    def __init__(self):
        super(refinementSub,self).__init__()

        self.concat = nn.Sequential(Concat())
        self.concat2 = nn.Sequential(Concat(1))
        #iteration1 --share
        #prepare input
        self.v_flow0 = nn.Sequential(Scale(1))
        self.compress_conv1a1b = nn.Sequential(conv(64, 16,3,1,1),
                                               nn.LeakyReLU(negative_slope=0.1))
        self.corr_min = nn.Sequential(corr(40, 1, 40, 1, 1))
        # iresnet
        ### itration 1 --changing
        self.FlowWarp0_itr1 = nn.Sequential(FlowWarp())
        self.upconv1a_sub_warped_upconv1b_itr1 = nn.Sequential(Eltwise())
        self.error_abs0_itr1 = nn.Sequential(AbsVal())
        ##begin
        #conv0
        self.ires_conv0_itr1 = nn.Sequential(conv(65, 32, 3, 1, 1),
                                             nn.LeakyReLU(negative_slope=0.1))
        #conv1
        self.ires_conv1_itr1 = nn.Sequential(conv(32, 64, 3, 2, 1),
                                             nn.LeakyReLU(negative_slope=0.1))
        #conv2
        self.ires_conv2_itr1 = nn.Sequential(conv(64, 128, 3, 2, 1),
                                             nn.LeakyReLU(negative_slope=0.1))
        #conv2b
        self.ires_conv2b_itr1 = nn.Sequential(conv(128, 128,3, 1, 1),
                                              nn.LeakyReLU(negative_slope=0.1))
        ###predict res2
        self.ires_Convolution2_itr1 = nn.Sequential(conv(128, 1, 3, 1, 1))
        self.ires_deconv2_itr1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                               nn.LeakyReLU(negative_slope=0.1))
        self.ires_upsample_2to1_itr1 = nn.Sequential(nn.ConvTranspose2d(1, 1, 4, 2, 1))
        ###predict res1
        self.ires_fused1_itr1 = nn.Sequential(conv(96, 64, 3, 1, 1))
        self.ires_convolution1_itr1 = nn.Sequential(conv(64, 1, 3, 1, 1))
        self.ires_deconv1_itr1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                               nn.LeakyReLU(negative_slope=0.1))
        self.ires_upsample_1to0_itr1 = nn.Sequential(nn.ConvTranspose2d(1, 1, 4, 2, 1))
        ###presict res0
        self.res_fused0_itr1 = nn.Sequential(conv(97, 32, 3, 1, 1))
        self.ires_Convolution0_itr1 = nn.Sequential(conv(32, 1, 3, 1, 1))
        self.sumation_initial_res0_itr1 = nn.Sequential(Eltwise(),
                                                        nn.LeakyReLU(negative_slope=0))

    def forward(self, x,conv1a, conv1b, up_conv1b2b):
        v_flow0 = self.v_flow0(x)
        conv1a_mini = self.compress_conv1a1b(conv1a)
        conv1b_mini = self.compress_conv1a1b(conv1b)
        corr_mini = self.corr_mini(conv1a_mini, conv1b_mini)
        # iresnet
        # itration 1 --changing
        disp2flow_final_itr1 = self.concat2([x, v_flow0])
        warp_error_itr1 = self.FlowWarp0_itr1(up_conv1b2b, disp2flow_final_itr1)
        warp_error_abs_itr1 = self.error_abs0_itr1(warp_error_itr1)
        ires_input_itr1 = self.concat2([up_conv1b2b, x, warp_error_abs_itr1])
        #begin?
        ires_conv0_itr1 = self.ires_conv0_itr1(ires_input_itr1)
        ires_conv1_itr1 = self.ires_conv1_itr1(ires_conv0_itr1)
        #*******
        concat_corr_itr1 = self.concat2([corr_mini, ires_conv1_itr1])
        #conv1b
        ires_conv1b_itr1 = self.ires_conv0_itr1(concat_corr_itr1)
        #conv2
        ires_conv2_itr1 = self.ires_conv2_itr1(ires_conv1b_itr1)
        ires_conv2b_itr1 = self.ires_conv2b_itr1(ires_conv2_itr1)
        #predict res2
        ires_predict2_res_itr1 = self.ires_Convolution2_itr1(ires_conv2b_itr1)
        ires_deconv2_itr1 = self.ires_deconv2_itr1(ires_conv2b_itr1)
        ires_upsampled_2to1_itr1 = self.ires_upsample_2to1_itr1(ires_predict2_res_itr1)
        ires_concat2_itr1 = self.concat([ires_conv1b_itr1, ires_deconv2_itr1])

        #predict res1
        ires_fused1_itr1 = self.ires_fused1_itr1(ires_concat2_itr1)
        ires_predict1_res_itr1 = self.ires_Convolution1_itr1(ires_fused1_itr1)
        ires_deconv1_itr1 = self.ires_deconv1_itr1(ires_fused1_itr1)
        ires_upsampled_1to0_itr1 = self.ires_upsample_1to0_itr1(ires_predict1_res_itr1)
        ires_concat1_itr1 = self.concat([ires_conv0_itr1, ires_deconv1_itr1, ires_upsampled_1to0_itr1])

        #predict res0
        ires_fused0_itr1 = self.ires_fused0_itr1(ires_concat1_itr1)
        ires_predict0_res_itr1 = self.ires_Convolution0_itr1(ires_fused0_itr1)
        ires_predict0_itr1 = self.sumation_initial_res0_itr1(x, ires_predict0_res_itr1)
        return ires_predict0_itr1