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
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--counter', dafault = 0, help='iteration times.')
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
        batch_size=2, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=2, shuffle=False, num_workers=4, drop_last=False)

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
        else:
            log.info("=> no checkpoint '{}'".format((args.loadmodel)))
            log.info("=>will start from scratch.")
    else:
        log.info("Not Resume")
        # train
        start_full_time = time.time()   #count the time training used
        for epoch in range(args.start_epoch, args.epoch):
            log.info('This is {}-th epoch'.format(epoch))

            train(train_left_img, train_right_img, test_left_disp,model, optimizer, log)

            savefilename = args.savepath + '/checkpoint.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
            savefilename)

        test(test_left_disp, test_right_img, test_left_disp, model, log);
        log.info('Full traing time = {:2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(imgL, imgR, disp_L, model, optimizer, log):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #------
        optimizer.zero_grad()

        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss= torch.nn.L1Loss(size_average=True, reduce=True)

        loss.backwoard()
        optimizer.step()

        return loss.data[0]

def test(imgL, imgR, disp_true, model, log):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    #----
    mask = disp_true < args.maxdisp
    #---

    with torch.no_grad():
        output = model(imgL, imgR)

    output = torch.squeeze(output.data.cpu(),1)[:,4:,:]

    if len(disp_true[mask]):
        EPE = 0
    else:
        EPE = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

    log.info('EPE='.format(EPE))


    return EPE