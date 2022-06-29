import sys
sys.path.append('../../')

import numpy as np
from moabb.datasets import BNCI2015004
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
import pickle
##########For cross subject -- more than one subject from source
# setup the paradigm
dataset =BNCI2015004()
dataset.subject_list = list(range(1, 10)) #
paradigm = MotorImagery(events = ['feet', 'navigation', 'right_hand', 'subtraction', 'word_ass'],n_classes=5)#
for i in range(1,10):#
    source = {}
    dataset_source = BNCI2015004()
    subject_source = [i]#
    X, labels, meta = paradigm.get_data(dataset_source, subjects=subject_source)
    source['org'] = {}

    source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X[:,2:,:])#
    source['org']['labels'] = labels
    #### the code to save the object
    ## saveing obejct
    var_name = 'BNCI2015004_' + str(i) + '_SPD'

    with open(var_name,'wb') as file1:
        pickle.dump(source,file1)


