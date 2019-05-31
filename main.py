from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models.IresNet import iresNet
import utils.logger as logger
from models import *

parser = argparse.ArgumentParser(description='IRESNet')
parser.add_argument('--maxdisp', type=int ,default=192, help='maxium disparity')
parser.add_argument('--datapath', default='dataset/', help='datapath')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--loadmodel', default= None, help='load model')
parser.add_argument('--savepath', default='./savedmodel', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def main():
    global args
    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()

    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=12, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))

    lr = args.lr
    model = iresNet()
    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0

    if args.savepath:
        if os.path.isfile(args.loadmodel):
            log.info("=> loading checkpoint '{}'".format((args.loadmodel)))
            checkpoint = torch.load(args.loadmodel)
            args.start_epoch = checkpoint['epoch']

