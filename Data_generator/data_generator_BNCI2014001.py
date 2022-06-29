"""

"""

import numpy as np
from torch.utils.data import Dataset
import moabb
from moabb.datasets import Zhou2016,BNCI2014001
from moabb.paradigms import LeftRightImagery,MotorImagery
import pickle


dataset = BNCI2014001()

dataset.subject_list = list(range(1, 10)) #total subjects=list(range(1, 10))#
paradigm = MotorImagery(events = ["left_hand", "right_hand", "feet", "tongue"],n_classes=4)#
#
for i in range(1,10):#
    train_x, train_y, _= paradigm.get_data(dataset=dataset, subjects=[i])# 训练样本list(range(1, 9))
    var_name = 'BNCI2014001_subject_' + str(i) + '_Trails'
    var_label = 'BNCI2014001_subject_' + str(i) + '_labels'
    with open(var_name, 'wb') as file1:
        pickle.dump(train_x, file1)
    with open(var_label, 'wb') as file1:
        pickle.dump(train_y, file1)


