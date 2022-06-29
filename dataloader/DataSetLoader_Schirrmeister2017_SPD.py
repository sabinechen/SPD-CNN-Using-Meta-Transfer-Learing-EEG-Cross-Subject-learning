

import numpy as np
from torch.utils.data import Dataset
import pickle

class DataSetLoader_Schirrmeister2017_SPD(Dataset):
    def __init__(self, setname, args, train_aug=False,TrainSubjects=[1,2],ValSubject=[3],TestSubject=[4],BinaryClassify = 0):
        ## loading All object
        RawData={}
        for i in range(1,15):#

            var_name = './dataloader/Schirrmeister2017_SPD/'+'Schirrmeister2017Subject_' + str(i)+'_SPD'

            with open(var_name,'rb') as file1:
                name = 'Schirrmeister2017Subject' + str(i) +'_Trails'
                SubjectData=pickle.load(file1)
                RawData[name]=SubjectData['org']['covs']
                name = 'Schirrmeister2017Subject' + str(i) + '_labels'
                RawData[name] = SubjectData['org']['labels']
        #Choose Subject
        train_x = None#
        subject_divide={}#
        for i in TrainSubjects:
            var_name = 'Schirrmeister2017Subject' + str(i) +'_Trails'
            var_label = 'Schirrmeister2017Subject' + str(i) + '_labels'
            if train_x is None:
                train_x = RawData[var_name]
                train_y = RawData[var_label]
            else:
                train_x = np.concatenate((train_x, RawData[var_name]), axis=0)
                train_y = np.concatenate((train_y, RawData[var_label]), axis=0)
            subject_divide[i] = len(train_x)  #
        self.sub_div = subject_divide
        test_x = None
        for i in TestSubject:
            var_name = 'Schirrmeister2017Subject' + str(i) +'_Trails'
            var_label = 'Schirrmeister2017Subject' + str(i) + '_labels'
            if test_x is None:
                test_x = RawData[var_name]
                test_y = RawData[var_label]
            else:
                test_x = np.concatenate((test_x, RawData[var_name]), axis=0)
                test_y = np.concatenate((test_y, RawData[var_label]), axis=0)
        val_x = None
        for i in ValSubject:
            var_name = 'Schirrmeister2017Subject' + str(i) +'_Trails'
            var_label = 'Schirrmeister2017Subject' + str(i) + '_labels'
            if val_x is None:
                val_x = RawData[var_name]
                val_y = RawData[var_label]
            else:
                val_x = np.concatenate((val_x, RawData[var_name]), axis=0)
                val_y = np.concatenate((val_y, RawData[var_label]), axis=0)
        del RawData
        ###### normalization
        i = 0
        for X in train_x:
            X_minMax = (X - X.mean()) / X.std()
            train_x[i, :, :] = X_minMax  #T
            i = i + 1
        i = 0
        for X in test_x:
            X_minMax = (X - X.mean()) / X.std()
            test_x[i, :, :] = X_minMax
            i = i + 1
        i = 0
        for X in val_x:
            X_minMax = (X - X.mean()) / X.std()
            val_x[i, :, :] = X_minMax
            i = i + 1

        #for pytorch
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        val_x = val_x.astype('float32')
        train_y[train_y == "right_hand"] = int(0)
        train_y[train_y == "left_hand"] = int(1)
        train_y[train_y == "rest"] = int(2)
        train_y[train_y == "feet"] = int(3)
        train_y = train_y.astype(int)
        test_y[test_y == "right_hand"] = int(0)  #
        test_y[test_y == "left_hand"] = int(1)
        test_y[test_y == "rest"] = int(2)
        test_y[test_y == "feet"] = int(3)
        test_y = test_y.astype(int)
        val_y[val_y == "right_hand"] = int(0)  #
        val_y[val_y== "left_hand"] = int(1)
        val_y[val_y == "rest"] = int(2)
        val_y[val_y== "feet"] = int(3)
        val_y = val_y.astype(int)

        ## for the input size of network
        classs_set =set(test_y)
        self.num_class = len(classs_set)
        self.in_chans=np.size(test_x,1)
        self.time_step=np.size(test_x,2)
        ###
        train_raw_x = np.transpose(train_x, [0, 2, 1])  #
        test_raw_x = np.transpose(test_x, [0, 2, 1])
        val_raw_x  = np.transpose(val_x, [0, 2, 1])


        #
        train_win_x = np.expand_dims(train_raw_x, axis=1)
        test_win_x = np.expand_dims(test_raw_x, axis=1)
        val_win_x  = np.expand_dims(val_raw_x, axis=1)
        train_win_y = train_y
        test_win_y = test_y
        val_win_y=val_y


        index = [i for i in range(len(test_win_x))]  #
        np.random.seed(12) #
        np.random.shuffle(index)  #
        test_win_x = test_win_x[index]
        test_win_y = test_win_y[index]

        index = [i for i in range(len(val_win_x))]  #
        np.random.seed(12) #
        np.random.shuffle(index)  #
        val_win_x = val_win_x[index]
        val_win_y = val_win_y[index]


        Number = np.size(test_win_x, 0)
        SampleNumber = int(Number * 1 / 12)
        self.X_test= test_win_x[:SampleNumber, :, :, :]  #
        self.y_test = test_win_y[:SampleNumber]

        Number = np.size(val_win_x, 0)
        SampleNumber = int(Number * 1/ 14)#
        self.X_val = val_win_x[:SampleNumber, :, :, :]  #
        self.y_val = val_win_y[:SampleNumber]

        if setname == 'train':
            self.data=train_win_x
            self.label=train_win_y
        elif setname == 'val':
            Number = np.size(val_win_x, 0)
            SampleNumber = int(Number * 1 / 5)  #
            self.data = val_win_x[:SampleNumber, :, :, :]  #
            self.label = val_win_y[:SampleNumber]
        elif setname == 'test':
            Number = np.size(test_win_x, 0)
            SampleNumber = int(Number * 1 / 3)
            self.data = test_win_x[:SampleNumber, :, :, :]  #
            self.label = test_win_y[:SampleNumber]
        print('End of Preparing')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i): #
        data, label=self.data[i], self.label[i]
        return data, label


