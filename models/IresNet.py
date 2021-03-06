from __future__ import print_function
from .correlation_package.correlation import Correlation as corr
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import skimage
from .layers import *
from .sub_nets import *

class iresNet(nn.Module):
    def __init__(self):
        super(iresNet, self).__init__()
        self.basisResNet = basisiResNet()
        self.refinementSub= refinementSub()
        self.prepareInput = prepareInput()
        self.Eltwise_disp_itr2 = nn.Sequential(Eltwise())


    def forward(self, leftImg,rightImg, disp_L):
        final_prediction,conv1a, conv1b, up_conv1b2b = self.basisResNet(leftImg, rightImg)
        v_flow0, corr_mini = self.prepareInput(final_prediction, conv1a, conv1b)
        ires_predict0_itr1 = self.refinementSub(final_prediction, v_flow0, corr_mini, up_conv1b2b)
        ires_predict0_itr2 = self.refinementSub(ires_predict0_itr1, v_flow0, corr_mini, up_conv1b2b)
        predict_disp_resize_itr2 = nn.Sequential(F.upsample(ires_predict0_itr2,mode='linear'))
        predict_disp_final_itr2 = Eltwise(predict_disp_resize_itr2)

        return predict_disp_final_itr2
