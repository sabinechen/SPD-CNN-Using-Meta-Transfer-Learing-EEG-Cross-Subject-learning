
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


class SPD_CNNnet(nn.Module):

    def __init__(self, in_chans=12, mtl=True):  #
        super(SPD_CNNnet, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        #
        # layer1-:
        self.conv1 = self.Conv2d(1, 4, (2, 2), padding=0)  #
        self.batchnorm1 = nn.BatchNorm2d(4, False)

        # Layer 2  #

        self.conv2 = self.Conv2d(4, 8, (2, 2), padding=0)
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d(2, 2)
        # Layer 3

        self.conv3 = self.Conv2d(8, 16, (3, 3))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling3 = nn.MaxPool2d(2, 2)

        #
        self.conv4 = self.Conv2d(16, 32, (3, 3))
        self.batchnorm4 = nn.BatchNorm2d(32, False)

        #
        self.conv5 = self.Conv2d(32, 64, (2, 2))
        self.batchnorm5 = nn.BatchNorm2d(64, False)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)

        # Layer 2
        # x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)

        # Layer 3
        # x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        #
        x = F.elu(self.conv4(x))
        x = self.batchnorm4(x)

        x = F.elu(self.conv5(x))
        x = self.batchnorm5(x)

        x = x.view(x.size(0),-1)  #
        return x


