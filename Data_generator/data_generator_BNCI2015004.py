"""

"""

import numpy as np
from torch.utils.data import Dataset
import moabb
from moabb.datasets import BNCI2015004
from moabb.paradigms import LeftRightImagery,MotorImagery
import pickle
### try loading

dataset = BNCI2015004()
dataset.subject_list = list(range(1, 10)) #total subjects=list(range(1, 10))
paradigm = MotorImagery(events = ['feet', 'navigation', 'right_hand', 'subtraction', 'word_ass'],n_classes=5)#
for i in range(1,10):#
    train_x, train_y, _= paradigm.get_data(dataset=dataset, subjects=[i])#
    var_name = 'BNCI2015004_subject_' + str(i) + '_Trails'
    var_label = 'BNCI2015004_subject_' + str(i) + '_labels'
    with open(var_name, 'wb') as file1:
        pickle.dump(train_x, file1)
    with open(var_label, 'wb') as file1:
        pickle.dump(train_y, file1)


