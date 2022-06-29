import sys
sys.path.append('../../')

import numpy as np
from moabb.datasets import  Schirrmeister2017
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
import pickle
##########For cross subject -- more than one subject from source
# setup the paradigm
dataset = Schirrmeister2017()
#subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9...14]
dataset.subject_list = list((range(1, 15))) #
#events=dict(right_hand=1, left_hand=2, rest=3, feet=4)
paradigm=MotorImagery(events = ['right_hand', 'left_hand', 'rest', 'feet'],n_classes=4) #

# get data from source
for i in range(1,15):#
    source = {}
    dataset_source =  Schirrmeister2017()
    subject_source = [i]#
    X, labels, meta = paradigm.get_data(dataset_source, subjects=subject_source)
    source['org'] = {}
    source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X[:,2:,:])#
    source['org']['labels'] = labels

    #### the code to save the object
    ## saveing obejct
    var_name = 'Schirrmeister2017Subject_' + str(i) + '_SPD'

    with open(var_name,'wb') as file1:
        pickle.dump(source,file1)



