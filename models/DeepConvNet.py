##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/aliasvishnu/EEGNet
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Feature Extractor """
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from models.conv2d_mtl import Conv2dMtl

class DeepConvNet(nn.Module):

    def __init__(self, mtl=True,in_chans=30 ):  #
        super(DeepConvNet, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        ###### Conv Pool Block 1######
        # conv_time layger
        self.conv1_1 = self.Conv2d(1, 25, (10, 1), stride=(1,1))  #
        conv_stride=1
        batch_norm=True#
        self.conv1_2 = self.Conv2d(25, 25, (1, in_chans),stride=(conv_stride, 1),bias=not batch_norm)#
        batch_norm_alpha=0.1
        self.batchnorm1 = nn.BatchNorm2d(25, momentum=batch_norm_alpha,affine=True,eps=1e-5)#
        # "pool"
        pool_time_length=3
        self.pooling1 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(3, 1),padding=0,dilation=1,ceil_mode=False)
        ###### Conv Pool Block 2######
        drop_prob=0.5
        self.drop=nn.Dropout(p=drop_prob)#
        #
        self.conv2=self.Conv2d(25, 50, (10, 1), stride=(1,1),padding=0,bias=not batch_norm)
        self.batchnorm2 = nn.BatchNorm2d(50, momentum=batch_norm_alpha,affine=True,eps=1e-5)#根
        self.pooling2 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(3, 1))
        ###### Conv Pool Block 3 ######
        drop_prob=0.5
        self.drop=nn.Dropout(p=drop_prob)#
        # conv_ layger
        self.conv3=self.Conv2d(50, 100, (10, 1), stride=(1,1),padding=0,bias=not batch_norm)
        self.batchnorm3 = nn.BatchNorm2d(100, momentum=batch_norm_alpha,affine=True,eps=1e-5)#根
        self.pooling3 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(3, 1))

        ###### Conv Pool Block 4 ######
        drop_prob = 0.5
        self.drop = nn.Dropout(p=drop_prob)  #
        # conv_ layger
        self.conv4 = self.Conv2d(100, 200, (10, 1), stride=(1, 1), padding=0, bias=not batch_norm)
        self.batchnorm4 = nn.BatchNorm2d(200, momentum=batch_norm_alpha, affine=True,eps=1e-5)  #
        self.pooling4 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(3, 1))
        ###initialize:
        torch.nn.init.xavier_uniform_(self.conv1_1.weight, gain=1)
        torch.nn.init.constant_(self.conv1_1.bias, 0)
        param_dict = dict(list(self.named_parameters()))#
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv{:d}.weight".format(block_nr)]
            torch.nn.init.xavier_uniform_(conv_weight, gain=1)
        for block_nr in range(1, 5):
            bnorm_weight = param_dict["batchnorm{:d}.weight".format(block_nr)]
            bnorm_bias = param_dict["batchnorm{:d}.bias".format(block_nr)]
            torch.nn.init.constant_(bnorm_weight, 1)
            torch.nn.init.constant_(bnorm_bias, 0)
        ##



    def forward(self, x):
        x = x.permute(0, 3, 2, 1)#swap dimensions
        #
        x=x.permute(0, 3, 2, 1)# transpose time to spat
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)#
        x = F.elu(x)
        x = self.pooling1(x)

        x =self.drop(x)
        x =self.conv2(x)
        x = self.batchnorm2(x)
        x= F.elu(x)
        x = self.pooling2(x)

        x =self.drop(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x=F.elu(x)
        x = self.pooling3(x)#

        x =self.drop(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x=F.elu(x)
        x = self.pooling4(x)

        return x

        
