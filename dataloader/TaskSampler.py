##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/Sha-Lab/FEAT
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Specific-subject task Sampler for dataloader. """
import torch
import numpy as np
import random # choose subject

class TaskTrainingSampler():
    """The class to generate  data"""
    def __init__(self, label, n_batch, n_cls, n_per,subject_divide):#subject_divide
        self.n_batch = n_batch
        self.n_cls = n_cls#
        self.n_per = n_per#self.args.shot + self.args.val_query
        self.subject_divide=subject_divide
        label = np.array(label)#
        self.m_ind = []
        for i in range(max(label) + 1):#
            ind = np.argwhere(label == i).reshape(-1)#
            self.m_ind.append(ind)#

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):#
            batch = []
            #
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:#select class randomly
                l = self.m_ind[c]#
                # select the subject randomly
                RandomSubject=random.sample(self.subject_divide.keys(), 1)
                RandomSubject=RandomSubject[0]
                sub_div = list(self.subject_divide.keys())# divdive the data of specific
                for index,subject in enumerate(sub_div):
                    if RandomSubject==subject:
                        SubjectIndex=index
                ###
                if RandomSubject==sub_div[0]:#
                    l1 = l[l<=self.subject_divide[RandomSubject]]
                else:
                     l1 = l[ (l>self.subject_divide[ sub_div[SubjectIndex-1] ]) & (l<=self.subject_divide[RandomSubject])]
                l1=torch.from_numpy(l1)
                pos = torch.randperm(len(l1))[:self.n_per] #
                batch.append(l1[pos])#
            batch = torch.stack(batch).t().reshape(-1)#
            yield batch


