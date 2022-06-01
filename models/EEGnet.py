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

class EEGnet(nn.Module):

    def __init__(self, mtl=True,in_chans=30):  #
        super(EEGnet, self).__init__()
        if mtl: ### if use SS operation:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d

        in_filter=1 #
        self.conv1 = self.Conv2d(in_filter, 16, (1, in_chans), padding=0)  #
        self.batchnorm1 = nn.BatchNorm2d(16, False)#
        # Layer 2   #
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = self.Conv2d(1, 4, (2, 32))#
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = self.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        x = x.contiguous().view(x.size(0), -1)
        return x

        
