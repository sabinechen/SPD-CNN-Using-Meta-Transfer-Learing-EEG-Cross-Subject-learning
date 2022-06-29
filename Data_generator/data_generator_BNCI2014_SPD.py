import sys
sys.path.append('../../../')

import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
import pickle
##########For cross subject -- more than one subject from source
# setup the paradigm
dataset = BNCI2014001()
dataset.subject_list = list(range(1, 10))
paradigm = MotorImagery(events = ["left_hand", "right_hand", "feet", "tongue"],n_classes=4)#

# set the weights for each class in the dataset
weights_classes = {}
weights_classes['feet'] = 1
weights_classes['right_hand'] = 1
weights_classes['left_hand'] = 1
# get data from source
for i in range(1,10):#
    source = {}
    dataset_source = BNCI2014001()
    subject_source = [i]#
    X, labels, meta = paradigm.get_data(dataset_source, subjects=subject_source)
    source['org'] = {}
    source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X[:,2:,:])#
    source['org']['labels'] = labels
    #### the code to save the object
    ## saveing obejct
    var_name = 'BNCI2014Subject_' + str(i) + '_SPD'
    with open(var_name,'wb') as file1:
        pickle.dump(source,file1)



