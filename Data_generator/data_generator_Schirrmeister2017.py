"""

"""

# import numpy as np

import moabb
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import LeftRightImagery,MotorImagery
import pickle


dataset = Schirrmeister2017()
#subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9...14]
dataset.subject_list = list((range(1, 15))) #total subjects=list(range(1, 10))#注意这里要选择要用的所有subject
#events=dict(right_hand=1, left_hand=2, rest=3, feet=4)
paradigm=MotorImagery(events = ['right_hand', 'left_hand', 'rest', 'feet'],n_classes=4) #

for i in range(1,15):#
    train_x, train_y, _= paradigm.get_data(dataset=dataset, subjects=[i])
    var_name = 'Schirrmeister2017_subject_' + str(i) + '_Trails'
    var_label = 'Schirrmeister2017_subject_' + str(i) + '_labels'
    with open(var_name, 'wb') as file1:
        pickle.dump(train_x, file1)
    with open(var_label, 'wb') as file1:
        pickle.dump(train_y, file1)
