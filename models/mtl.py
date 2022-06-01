##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.EEGnet import EEGnet
from models.DeepConvNet import DeepConvNet
from models.ConvClassifier import ConvClassifier
from models.SPD_CNNnet import SPD_CNNnet
from utils.util import np_to_var
class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        layer=self.args.num_cls_lay
        hiddenlayer=self.args.num_cls_hidden
        if  layer==1:
            self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
            torch.nn.init.kaiming_normal_(self.fc1_w)#initialize method
            self.vars.append(self.fc1_w)
            self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
            self.vars.append(self.fc1_b)
        else:
            self.fc1_w = nn.Parameter(torch.ones([hiddenlayer, self.z_dim]))
            torch.nn.init.kaiming_normal_(self.fc1_w)#
            self.vars.append(self.fc1_w)
            self.fc1_b = nn.Parameter(torch.zeros(hiddenlayer))
            self.vars.append(self.fc1_b)

            self.fc2_w = nn.Parameter(torch.ones([self.args.way, hiddenlayer]))
            torch.nn.init.kaiming_normal_(self.fc2_w)#
            self.vars.append(self.fc2_w)
            self.fc2_b = nn.Parameter(torch.zeros(self.args.way))
            self.vars.append(self.fc2_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        layer=self.args.num_cls_lay
        if self.args.model_type=='Deep4':#
            input_x = input_x.view(input_x.size(0),-1)  #
        if layer==1:
            fc1_w = the_vars[0]
            fc1_b = the_vars[1]
            net = F.linear(input_x, fc1_w, fc1_b)
        else:
            fc1_w = the_vars[0]
            fc1_b = the_vars[1]
            fc2_w = the_vars[2]
            fc2_b = the_vars[3]
            input_x = F.linear(input_x, fc1_w, fc1_b)   #
            net = F.linear(input_x, fc2_w, fc2_b)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=4,in_chans=30,input_time_length=1793):
        super().__init__()
        self.args = args
        self.mode = mode
        self.model_type=args.model_type
        self.update_lr = args.base_lr
        self.update_step = args.update_step

        if self.model_type=="EEGNet":
            if self.mode == 'meta':  #
                self.encoder = EEGnet(in_chans = in_chans,mtl=args.MTL)#if args.MTL=false，then use the MAML without SS
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch,1, input_time_length,in_chans),dtype=np.float32)))
                final_layer_length = out.cpu().data.numpy().shape[1]
            else:
                self.encoder = EEGnet(in_chans = in_chans,mtl=False)
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch,1, input_time_length,in_chans),dtype=np.float32)))
                final_layer_length = out.cpu().data.numpy().shape[1]
                self.classifier = nn.Sequential(nn.Linear(final_layer_length, num_cls))
        elif self.model_type=='Deep4':
            if self.mode == 'meta': #
                self.encoder = DeepConvNet(in_chans=in_chans,mtl=args.MTL)
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch, 1, input_time_length, in_chans),dtype=np.float32)))
                out= out.view(out.size(0),-1)#
                final_layer_length=out.shape[1]
            else:
                self.encoder = DeepConvNet(in_chans=in_chans,mtl=False)
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch, 1, input_time_length, in_chans),dtype=np.float32)))
                final_layer_length= out.cpu().data.numpy().shape[2]
                self.classifier = ConvClassifier(mtl=False,n_classes=num_cls,final_conv_length=final_layer_length )
                out= out.view(out.size(0),-1)#
                final_layer_length=out.shape[1]
        elif self.model_type=='SPD_CNNnet':
            if self.mode == 'meta':
                self.encoder = SPD_CNNnet(in_chans=in_chans,mtl=args.MTL)
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch,1, input_time_length,in_chans),dtype=np.float32)))
                final_layer_length = out.cpu().data.numpy().shape[1]
            else:
                self.encoder = SPD_CNNnet(in_chans=in_chans,mtl=False)
                self.encoder.eval()
                out = self.encoder(np_to_var(np.ones((args.num_batch,1, input_time_length,in_chans),dtype=np.float32)))
                final_layer_length = out.cpu().data.numpy().shape[1]#
                self.classifier = nn.Sequential(nn.Linear(final_layer_length , num_cls))
        else:
            print("wrong input of ")#
            assert print("wrong input of NetworkStructure")
        self.final_layer_length =final_layer_length
        self.base_learner = BaseLearner(args, z_dim=self.final_layer_length)
    def forward(self, inp):
        if self.mode=='pre' or self.mode=='origval':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):

        return self.classifier(self.encoder(inp))


    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        self.base_learner = BaseLearner(self.args, z_dim=self.final_layer_length) #re-initialize the parameter of classifier block
        # Set base_learner to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.base_learner = self.base_learner.cuda()
        #
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params,lr=self.args.base_lr) #
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)#
        optimizer.step()
        logits_q = self.base_learner(embedding_query)
        for _ in range(self.update_step):  #
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True) #
            optimizer.step()
            logits_q = self.base_learner(embedding_query)
        return logits_q

    def meta_forward(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        self.base_learner = BaseLearner(self.args, z_dim=self.final_layer_length)#TODO:改正之后,这里重新定义了一一遍，防止每个epoch记住一次数据
        # Set base_learner to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.base_learner = self.base_learner.cuda()
        #
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params,lr=self.args.base_lr) #直接用adam，这里用默认的参数
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)#这里参数表明保留backward后的中间参数。
        optimizer.step()
        logits_q = self.base_learner(embedding_query)

        for _ in range(self.update_step):  #直接用训练十代(self.update_step=10)，不用手动赋值过程，直接用原始的adam训练,这里就是innerLoop的过程
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True) #这里参数表明保留backward后的中间参数。
            # loss.backward#好像没什么不同
            optimizer.step()
            logits_q = self.base_learner(embedding_query)

        return logits_q


