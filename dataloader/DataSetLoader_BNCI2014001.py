"""

"""

import numpy as np
import pickle
from torch.utils.data import Dataset

class DataSetLoader_BNCI2014001(Dataset):
    def __init__(self, setname, args, train_aug=False,TrainSubjects=[1,2,3,4,5,6],ValSubject=[7,8],TestSubject=[9], BinaryClassify =True):#
        ## loading object
        RawData={}
        for i in range(1,10):# B
            # 9subjects
            var_name = './dataloader/BNCI2014001/'+'BNCI2014001_subject_' + str(i) + '_Trails'
            var_label = './dataloader/BNCI2014001/'+'BNCI2014001_subject_' + str(i) + '_labels'
            with open(var_name,'rb') as file1:
                name = 'BNCI2014001_subject_' + str(i) + '_Trails'
                RawData[name]=pickle.load(file1)
            with open(var_label, 'rb') as file1:
                name = 'BNCI2014001_subject_' + str(i) + '_labels'
                RawData[name] = pickle.load(file1)
        #choose train subject and validation subject
        # TrainSubjects=[1,2,3,4,5,6,7,8]
        train_x= None
        subject_divide={}#
        for i in TrainSubjects:
            var_name = 'BNCI2014001_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2014001_subject_' + str(i) + '_labels'
            if train_x is None:
                train_x = RawData[var_name]
                train_y= RawData[var_label]
            else:
                train_x=np.concatenate((train_x,RawData[var_name]),axis=0)
                train_y = np.concatenate((train_y, RawData[var_label]), axis=0)
            subject_divide[i]=len(train_x)#  the data divide of different subject,used in meta_update phase
        self.sub_div=subject_divide
        test_x= None
        for i in TestSubject:
            var_name = 'BNCI2014001_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2014001_subject_' + str(i) + '_labels'
            if test_x is None:
                test_x = RawData[var_name]
                test_y= RawData[var_label]
            else:
                test_x=np.concatenate((test_x,RawData[var_name]),axis=0)
                test_y = np.concatenate((test_y, RawData[var_label]), axis=0)
        val_x = None
        for i in ValSubject:
            var_name = 'BNCI2014001_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2014001_subject_' + str(i) + '_labels'
            if val_x is None:
                val_x = RawData[var_name]
                val_y = RawData[var_label]
            else:
                val_x = np.concatenate((val_x, RawData[var_name]), axis=0)
                val_y = np.concatenate((val_y, RawData[var_label]), axis=0)

        del RawData
###

        train_x = train_x.astype('float32')# for pytorch
        test_x = test_x.astype('float32')
        val_x = val_x.astype('float32')
        #events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
        train_y[train_y == "left_hand"] = int(0)  #
        train_y[train_y == "right_hand"] = int(1)
        train_y[train_y == "feet"] = int(2)
        train_y[train_y == "tongue"] = int(3)
        train_y = train_y.astype(int)
        test_y[test_y == "left_hand"] = int(0)  #
        test_y[test_y == "right_hand"] = int(1)
        test_y[test_y == "feet"] = int(2)
        test_y[test_y == "tongue"] = int(3)
        test_y = test_y.astype(int)
        val_y[val_y == "left_hand"] = int(0)  #
        val_y[val_y== "right_hand"] = int(1)
        val_y[val_y == "feet"] = int(2)
        val_y[val_y== "tongue"] = int(3)
        val_y = val_y.astype(int)

        ##for the network input number
        classs_set =set(test_y)
        self.num_class = len(classs_set)
        self.in_chans=np.size(test_x,1)
        self.time_step=np.size(test_x,2)
        ###
        train_raw_x = np.transpose(train_x, [0, 2, 1])
        test_raw_x = np.transpose(test_x, [0, 2, 1])
        val_raw_x  = np.transpose(val_x, [0, 2, 1])


        #
        train_win_x = np.expand_dims(train_raw_x, axis=1)
        test_win_x = np.expand_dims(test_raw_x, axis=1)
        val_win_x  = np.expand_dims(val_raw_x, axis=1)
        train_win_y = train_y
        test_win_y = test_y
        val_win_y=val_y


        ##The user or target domain is  scrambled and  divided to prevent the test set and validation set comes from  different sessions
        index = [i for i in range(len(test_win_x))]
        np.random.seed(12) #
        np.random.shuffle(index)  #
        test_win_x = test_win_x[index]
        test_win_y = test_win_y[index]

        index = [i for i in range(len(val_win_x))]
        np.random.seed(12) #
        np.random.shuffle(index)  #
        val_win_x = val_win_x[index]
        val_win_y = val_win_y[index]



        #for original ML testing
        self.X_val=val_win_x#
        self.y_val=val_win_y
        self.X_test=test_win_x#
        self.y_test = test_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = test_win_x
            self.label = test_win_y



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i): #
        data, label=self.data[i], self.label[i]
        return data, label
