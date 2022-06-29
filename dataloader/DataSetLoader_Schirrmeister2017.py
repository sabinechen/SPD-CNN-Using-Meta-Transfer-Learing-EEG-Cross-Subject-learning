

import numpy as np
import pickle
from torch.utils.data import Dataset
# Schirrmeister2017
class DataSetLoader_Schirrmeister2017(Dataset):
    def __init__(self, setname, args, train_aug=False,TrainSubjects=[1,2,3,4,5,6],ValSubject=[7,8],TestSubject=[9], BinaryClassify =True):#选定测试subject与训练subjects
        ## loading object 采用字典的方式进行加载 #例如TrainSubjects=[1,2,3,4,5,6,7,8],TestSubject=[9]
        # dataset.subject_list = list((range(1, 15)))--14个subject
        RawData={}
        Allsubject=TrainSubjects+ValSubject+TestSubject
        # for i in range(1,15):# B
        for i in Allsubject:  # B
            # 14subjects
            var_name = './dataloader/Schirrmeister2017/'+'Schirrmeister2017_subject_' + str(i) + '_Trails'
            var_label = './dataloader/Schirrmeister2017/'+'Schirrmeister2017_subject_' + str(i) + '_labels'
            with open(var_name,'rb') as file1:
                name = 'Schirrmeister2017_subject_' + str(i) + '_Trails'
                RawData[name]=pickle.load(file1)
            with open(var_label, 'rb') as file1:
                name = 'Schirrmeister2017_subject_' + str(i) + '_labels'
                RawData[name] = pickle.load(file1)
        #选定测试subject与训练subjects # TODO:取出没一个subject的标号位置--后面就不需要再打乱了
        # TrainSubjects=[1,2,3,4,5,6,7,8]
        train_x= None
        subject_divide={}#取出没一个subject的标号以及名称
        for i in TrainSubjects:
            var_name = 'Schirrmeister2017_subject_' + str(i) + '_Trails'
            var_label = 'Schirrmeister2017_subject_' + str(i) + '_labels'
            if train_x is None:
                train_x = RawData[var_name]
                train_y= RawData[var_label]
            else:
                train_x=np.concatenate((train_x,RawData[var_name]),axis=0)
                train_y = np.concatenate((train_y, RawData[var_label]), axis=0)
            subject_divide[i]=len(train_x)# 各个subject的标号
        self.sub_div=subject_divide
        #提到前面，防止内存占用
        train_x = train_x.astype('float32')# pytorch need it
        train_raw_x = np.transpose(train_x, [0, 2, 1])  # channel 和 data调换位置
        del train_x

        test_x= None
        for i in TestSubject:
            var_name = 'Schirrmeister2017_subject_' + str(i) + '_Trails'
            var_label = 'Schirrmeister2017_subject_' + str(i) + '_labels'
            if test_x is None:
                test_x = RawData[var_name]
                test_y= RawData[var_label]
            else:
                test_x=np.concatenate((test_x,RawData[var_name]),axis=0)
                test_y = np.concatenate((test_y, RawData[var_label]), axis=0)
        ## for network input number
        test_x = test_x.astype('float32')
        test_raw_x = np.transpose(test_x, [0, 2, 1])

        classs_set =set(test_y)
        self.num_class = len(classs_set)  #
        self.in_chans=np.size(test_x,1)
        self.time_step=np.size(test_x,2)
        del test_x

        val_x = None
        for i in ValSubject:
            var_name = 'Schirrmeister2017_subject_' + str(i) + '_Trails'
            var_label = 'Schirrmeister2017_subject_' + str(i) + '_labels'
            if val_x is None:
                val_x = RawData[var_name]
                val_y = RawData[var_label]
            else:
                val_x = np.concatenate((val_x, RawData[var_name]), axis=0)
                val_y = np.concatenate((val_y, RawData[var_label]), axis=0)
        val_x = val_x.astype('float32')
        val_raw_x  = np.transpose(val_x, [0, 2, 1])
        del val_x

        del RawData
###

        train_y[train_y == "right_hand"] = int(0)  #
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


        #
        train_win_x = np.expand_dims(train_raw_x, axis=1)
        test_win_x = np.expand_dims(test_raw_x, axis=1)
        val_win_x  = np.expand_dims(val_raw_x, axis=1)
        train_win_y = train_y
        test_win_y = test_y
        val_win_y=val_y

        ##he user or target domain is  scrambled and  divided to prevent the test set and validation set comes from  different sessions
        index = [i for i in range(len(test_win_x))]  # test_data
        np.random.shuffle(index)  #
        test_win_x = test_win_x[index]
        test_win_y = test_win_y[index]

        index = [i for i in range(len(val_win_x))]
        np.random.shuffle(index)  #
        val_win_x = val_win_x[index]
        val_win_y = val_win_y[index]



        #decrease validation and test time
        Number = np.size(test_win_x, 0)
        SampleNumber = int(Number * 1 / 9)
        self.X_test= test_win_x[:SampleNumber, :, :, :]  #
        self.y_test = test_win_y[:SampleNumber]

        Number = np.size(val_win_x, 0)
        SampleNumber = int(Number * 1/ 15)#
        self.X_val = val_win_x[:SampleNumber, :, :, :]  #
        self.y_val = val_win_y[:SampleNumber]



        if setname == 'train':

            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            Number = np.size(val_win_x, 0)
            SampleNumber = int(Number * 1 / 12)  #
            self.data = val_win_x[:SampleNumber, :, :, :]  #
            self.label = val_win_y[:SampleNumber]
            # self.data = val_win_x
            # self.label = val_win_y
        elif setname == 'test':
            Number = np.size(test_win_x, 0)
            SampleNumber = int(Number * 1 / 3)
            self.data = test_win_x[:SampleNumber, :, :, :]  #
            self.label = test_win_y[:SampleNumber]




    def __len__(self):
        return len(self.data)

    def __getitem__(self, i): #
        data, label=self.data[i], self.label[i]
        return data, label
