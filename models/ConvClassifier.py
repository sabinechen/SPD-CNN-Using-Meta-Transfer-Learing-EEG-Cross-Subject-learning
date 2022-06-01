##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

class ConvClassifier(nn.Module):

    def __init__(self, mtl=True,n_classes=2,final_conv_length=17):  #
        super(ConvClassifier, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d

        ### Conv classifier
        n_cls=n_classes
        self.conv_classifier = self.Conv2d(200, n_cls, (final_conv_length, 1), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        nn.init.constant_(self.conv_classifier.bias, 0)

    def forward(self, x):

        x=self.conv_classifier(x)
        x= F.log_softmax(x, 1, _stacklevel=5)
        def squeeze_final_output(x):
            assert x.size()[3] == 1
            x = x[:, :, :, 0]
            if x.size()[2] == 1:
                x = x[:, :, 0]
            return x
        x= squeeze_final_output(x)

        return x


