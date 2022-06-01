"""

"""

import numpy as np
import pickle
from torch.utils.data import Dataset

class DataSetLoader_BNCI2015004(Dataset):
    def __init__(self, setname, args, train_aug=False,TrainSubjects=[1,2,3,4,5,6,7],ValSubject=[7,8],TestSubject=[9],BinaryClassify = 0):#选定测试subject与训练subjects
        ## loading object 采用字典的方式进行加载
        RawData={}
        for i in range(1,10):# B
            # 9subjects
            var_name = './dataloader/BNCI2015004/'+'BNCI2015004_subject_' + str(i) + '_Trails'
            var_label = './dataloader/BNCI2015004/'+'BNCI2015004_subject_' + str(i) + '_labels'
            with open(var_name,'rb') as file1:
                name = 'BNCI2015004_subject_' + str(i) + '_Trails'
                RawData[name]=pickle.load(file1)
            with open(var_label, 'rb') as file1:
                name = 'BNCI2015004_subject_' + str(i) + '_labels'
                RawData[name] = pickle.load(file1)
        #选定测试subject与训练subjects
        # TrainSubjects=[1,2,3,4,5,6,7,8]
        train_x= None
        subject_divide={}#取出没一个subject的标号以及名称
        for i in TrainSubjects:
            var_name = 'BNCI2015004_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2015004_subject_' + str(i) + '_labels'
            if train_x is None:
                train_x = RawData[var_name]
                train_y= RawData[var_label]
            else:
                train_x=np.concatenate((train_x,RawData[var_name]),axis=0)
                train_y = np.concatenate((train_y, RawData[var_label]), axis=0)
            subject_divide[i]=len(train_x)# 各个subject的标号
        self.sub_div=subject_divide
        # TestSubject=[9]
        test_x= None
        for i in TestSubject:
            var_name = 'BNCI2015004_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2015004_subject_' + str(i) + '_labels'
            if test_x is None:
                test_x = RawData[var_name]
                test_y= RawData[var_label]
            else:
                test_x=np.concatenate((test_x,RawData[var_name]),axis=0)
                test_y = np.concatenate((test_y, RawData[var_label]), axis=0)
        val_x = None
        for i in ValSubject:
            var_name = 'BNCI2015004_subject_' + str(i) + '_Trails'
            var_label = 'BNCI2015004_subject_' + str(i) + '_labels'
            if val_x is None:
                val_x = RawData[var_name]
                val_y = RawData[var_label]
            else:
                val_x = np.concatenate((val_x, RawData[var_name]), axis=0)
                val_y = np.concatenate((val_y, RawData[var_label]), axis=0)
        del RawData
###

        # 将字符标签进行替换
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        val_x = val_x.astype('float32')
        #这里不知道为什么进去的还是字符类型,也许是因为ndarry不能存储不同类型的数据，全部当成字符存起来再该
        train_y[train_y == 'word_ass'] = int(0)  #  right_hand=4, feet=5, navigation=3, subtraction=2, word_ass=1 # 索引如果不从零开始会出问题,这里全部减去1
        train_y[train_y == 'subtraction'] = int(1)
        train_y[train_y == 'navigation'] = int(2)
        train_y[train_y == 'right_hand'] = int(3)
        train_y[train_y == 'feet'] = int(4)
        train_y = train_y.astype(int)
        test_y[test_y == 'word_ass'] = int(0)  #  转换后word_ass=0 subtraction=1 navigation=2 right_hand=3 feet=4,
        test_y[test_y == 'subtraction'] = int(1)
        test_y[test_y == 'navigation'] = int(2)
        test_y[test_y == 'right_hand'] = int(3)
        test_y[test_y == 'feet'] = int(4)
        test_y = test_y.astype(int)
        val_y[val_y == 'word_ass'] = int(0)  #  转换后word_ass=0 subtraction=1 navigation=2 right_hand=3 feet=4,
        val_y[val_y== 'subtraction'] = int(1)
        val_y[val_y == 'navigation'] = int(2)
        val_y[val_y== 'right_hand'] = int(3)
        val_y[val_y== 'feet'] = int(4)
        val_y = val_y.astype(int)
        # train_y-=1 #减去1 估计让0作为开头 #TODO:索引如果不从零开始pytorch会出问题,pytoch的机制
        # test_y-=1
        ### TODO: 拟写一段代码，筛选需要的类别并且统计数量 ##### 这里例子是筛选出0 1 的类别(i==0 or i==1), 注意这里要是选其中两类出来要讲他们置换成 0 1 ，因为pytorch标签需要重0-1开始
        # BinaryClassify = True
        if BinaryClassify==True:#  Todo:注意这里不适用最新的task-trainning
            classifyClass=[3,4] #选两个类别进行实验
            ClassIndex=[index for index,i in enumerate(train_y) if i==classifyClass[0] or i==classifyClass[1]]#注意index Python里面是0开头
            train_x=train_x[ClassIndex]
            train_y=train_y[ClassIndex]
            ##因为pytorch标签需要重0-1开始
            train_y[train_y==classifyClass[0]]=int(0)#FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`.
            train_y[train_y == classifyClass[1]] = int(1)

            ClassIndex=[index for index,i in enumerate(test_y) if i==classifyClass[0] or i==classifyClass[1]]#注意index Python里面是0开头
            test_x=test_x[ClassIndex]
            test_y=test_y[ClassIndex]
            test_y[test_y==classifyClass[0]]=int(0)
            test_y[test_y == classifyClass[1]]= int(1)

            ClassIndex=[index for index,i in enumerate(val_y) if i==classifyClass[0] or i==classifyClass[1]]#注意index Python里面是0开头
            val_x=val_x[ClassIndex]
            val_y=val_y[ClassIndex]
            ##因为pytorch标签需要重0-1开始
            val_y[val_y==classifyClass[0]]=int(0)#FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`.
            val_y[val_y == classifyClass[1]] = int(1)

        ## 需要对外传出的参数, to change the network input number
        classs_set =set(test_y)
        self.num_class = len(classs_set)  # 该数据集合 就左手 右手 和脚三类 ,不知道为什么只两类---现在好了能因为BOABB不熟悉导致的
        self.in_chans=np.size(test_x,1)
        self.time_step=np.size(test_x,2)
        ###
        train_raw_x = np.transpose(train_x, [0, 2, 1])  # channel 和 data调换位置
        test_raw_x = np.transpose(test_x, [0, 2, 1])
        val_raw_x  = np.transpose(val_x, [0, 2, 1])

        # 这里直接当成一个维度训练一遍。先不用窗口的方法,这里相当与窗口大小就是整个的大小,但问题可能就是每个大小都不一样那就GG（不对已经切过了，肯定都一样）
        train_win_x = np.expand_dims(train_raw_x, axis=1)
        test_win_x = np.expand_dims(test_raw_x, axis=1)
        val_win_x  = np.expand_dims(val_raw_x, axis=1)
        train_win_y = train_y
        test_win_y = test_y
        val_win_y=val_y
        ### 判断是不是受试者实验
        if TrainSubjects==TestSubject:#TODO:拟用统计法统计样本总数len等，乘上相应的比例，比如 train:val:test=3:1:1，然后在pre阶段加入orginal_test阶段来做内试者实验
            SampleNamber=np.size(train_win_x, 0)
            TrainWeight=int(SampleNamber*3/5)
            ValWeight=int(SampleNamber*1/5)
            TestWeight=int(SampleNamber*1/5)#划分验证集合大小
            train_win_x=train_win_x[:TrainWeight, :, :, :] # 这样自己会覆盖自己缩小值域
            train_win_y= train_win_y[:TrainWeight]

            val_win_x = test_win_x[TrainWeight:TrainWeight+ValWeight, :, :, :]#因为这里TrainSubjects==TestSubject
            val_win_y = test_win_y[TrainWeight:TrainWeight+ValWeight]

            test_win_x = test_win_x[TrainWeight+ValWeight:TrainWeight+ValWeight+TestWeight, :, :, :]
            test_win_y =test_win_y[TrainWeight+ValWeight:TrainWeight+ValWeight+TestWeight]
        else:# 则为cross-subject
            ##先将“使用者” 或者说target domain打乱再进行划分，防止测试集和验证集合取得不同session
            index = [i for i in range(len(test_win_x))]  # test_data为测试数据
            np.random.seed(12) #固定生成随机数的序列，免得下次读取又打乱一次，混淆val 与 test数据
            np.random.shuffle(index)  # 打乱索引
            test_win_x = test_win_x[index]
            test_win_y = test_win_y[index]

            index = [i for i in range(len(val_win_x))]  # test_data为测试数据
            np.random.seed(12) #固定生成随机数的序列，免得下次读取又打乱一次，混淆val 与 test数据
            np.random.shuffle(index)  # 打乱索引
            val_win_x = val_win_x[index]
            val_win_y = val_win_y[index]

        #原始参数传递方式
        self.X_val=val_win_x# train一个epoch时候执行 原始validate阶段时候要用,用原始传参数的方式传出去，不用到torch里面的及价值
        self.y_val=val_win_y
        self.X_test=test_win_x#用于original_test 阶段
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

    def __getitem__(self, i): #最后用到win数据，就是原来一段eeg按step切成七片对应用一个样本标号，也就是可以扩充sample数量
        data, label=self.data[i], self.label[i]
        return data, label
