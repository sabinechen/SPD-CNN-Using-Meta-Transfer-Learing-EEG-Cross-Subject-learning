# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta_update import MetaTrainer  #
from trainer.pre import PreTrainer
from trainer.TraditionalTest import TestModel
import time

if __name__ == '__main__':
    start = time.time()  # calculate time
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='EEGNet', choices=['EEGNet'])  # The network architecture
    parser.add_argument('--dataset', type=str, default='BNCI2015004')  # Dataset
    parser.add_argument('--P300', type=int, default=0)  # if P300=1 ,else==0 MI etc==0
    parser.add_argument('--MTL', type=int, default=1)  # if MTL=1 ,(MAML) MTL=0
    # parser.add_argument('--n_classes', type=int, default=5)  # base on the dataset you use
    parser.add_argument('--TrainSubjects', nargs='+')
    parser.add_argument('--TestSubject', nargs='+')
    parser.add_argument('--ValSubject', nargs='+')
    # To make the input integers#貌似多了这个限定就可以引入数组参数
    parser.add_argument('--TrainSubjects-int-type', nargs='+', type=int)
    parser.add_argument('--TestSubject-int-type', nargs='+', type=int)
    parser.add_argument('--ValSubject-int-type', nargs='+', type=int)
    parser.add_argument('--phase', type=str, default='meta_train',choices=['pre_train', 'meta_train', 'meta_eval'])  # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed # 可与复现
    parser.add_argument('--gpu', default='0')  # GPU id
    # parser.add_argument('--dataset_dir', type=str,default='./data/')  # Dataset folder
    # # z_dim --The number for neural after the flatten of encoder
    # parser.add_argument('--z_dim', type=int, default=4*2*112)
    # if binary classification
    parser.add_argument('--BinaryClassify', type=int, default=0)  #

    # Parameters for meta-train and meta-validation phase
    # Epoch number for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=12)    # The number for different tasks used for meta-train
    parser.add_argument('--num_batch', type=int, default=20)    # meta-batch size : like 4 task into 1 meta-batch ,20task into 5 meta task
    parser.add_argument('--meta_batch_size', type=int, default=4)
    parser.add_argument('--shot', type=int, default=5)    # Shot number, how many samples for one class in a task-ol
    parser.add_argument('--way', type=int, default=3)    # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=10)    # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=10)  #  # The number of test samples for each class in a task
    # Learning rate for SS weights   #the lr1-2 are put into adam opti
    parser.add_argument('--meta_lr1', type=float, default=0.0001)    # Learning rate for FC weights
    parser.add_argument('--meta_lr2', type=float, default=0.005)    # Learning rate for the inner loop
    parser.add_argument('--base_lr', type=float,default=0.005)  #
    parser.add_argument('--update_step', type=int, default=100)   #The number of updates for the inner loop

    parser.add_argument('--step_size', type=int, default=3)    # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.8)    # Gamma for the meta-train learning rate decay
    ## base-layer
    parser.add_argument('--num_cls_lay', type=int, default=2)
    parser.add_argument('--num_cls_hidden', type=int, default=32)  #
    #####################3
    # The pre-trained weights for meta-train phase
    parser.add_argument('--init_weights', type=str, default=None)  #
    # The meta-trained weights for meta-eval phase
    parser.add_argument('--eval_weights', type=str,default=None)  #  Load model for meta-test phase
    # Additional label for meta-train
    parser.add_argument('--meta_label', type=str, default='2021111601')  # label

    # Parameters for pretain phase
    # Epoch number for pre-train phase
    parser.add_argument('--pre_max_epoch', type=int, default=30)
    # Batch size for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=12)
    # # embedding size
    # Learning rate for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=1e-3)  #
    #####################3
    # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_gamma', type=float, default=0.5)  #
    # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_step_size', type=int, default=20)  #
    # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9)  #
    #####################3
    # Weight decay for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005)  #
    # Additional label for pre-train
    parser.add_argument('--pre_train_label', type=str, default='2021111601')  # label for date

    # Set the parameters
    #####  base parameter  ####
    args.gpu = '2'
    args.phase = 'pre_train'
    args.pre_batch_size = 64  # 听说64适用于大部分脑电分类

    # Choose the net work structrue and datset
    # args.model_type="EEGNet"
    # args.model_type="Deep4"
    args.model_type = "SPD_CNNnet"
    args.TrainSubjects = [1,2,5,8,10,12,13,14]  ##选定测试subject与训练subjects 当选定为同一个时候触发intersubject学习,这个需要根据数据集合来选择
    args.ValSubject = [4,9,11]
    args.TestSubject = [7,8]  # 开始debug

    # args.dataset='BNCI2014001'# 若在meta传参数还得想想办法
    # args.dataset='BNCI2015004'# 若在meta传参数还得想想办法
    args.dataset='Schirrmeister2017_SPD'# 若在meta传参数还得想想办法

    args.way =  4# 只要少于总的类别数目就可以，一般这里直接用少于脑电分类总的类别数目，减低难度
    args.BinaryClassify = 0  # 多分类(00)还是二分类(=1)
    # args.BinaryClassify=1    # 多分类(00)还是二分类(=1)

    #### The parameter of pretrain (based on the parameter from run_pre.py:
    args.pre_max_epoch = 60  # 10epoch ，因为样本特征量没这么大？  一般可以train多几个epoch，因为pre-train中的overfitting 对最后性能影响不大
    args.pre_train_label = 20220316030
    # For lr 变速learning rate
    args.pre_lr = 1e-3  # adam 默认是1e-3  # 根据原始mtl 可以适当增大
    args.pre_gamma = 0.9
    args.pre_step_size = 40  #

    ####meta-val和pre-val阶段
    args.shot = 10  # pre train 用于 meta val阶段，设置大一点有利于选出泛化能力强一点的pretrain模型？
    args.train_query = 10  # pre train 用于 meta val阶段，设置大一点有利于选出泛化能力强一点的pretrain
    args.base_lr = 1e-3  # 这个是用于adaption更新阶段手写梯度时候用来更新的，这里传入用adam优化中，默认1e03
    args.update_step = 75  # 用于adaption小更新train多少代,
    args.num_cls_lay = 1  # 默认两层全连接层作为
    args.num_cls_hidden = 32  # 分类层隐含层的信息
    ## meta train
    args.gamma = 0.9
    args.step_size = 5  # 3epoch之后
    args.meta_lr1 = 0.001  # For encoder(SS) weight
    args.meta_lr2 = 0.001  # For classier(FCweight)
    args.num_batch = 60  # # The number for different tasks used for meta-train
    args.meta_batch_size = 5  # 需要是num_batch公倍数
    args.max_epoch = 40  # 训练多少代
    # args.meta_label=20220117000
    args.meta_label = args.pre_train_label  # 暂时设置相同

    #### _____________________________________________End of Debug_______________________________________________
    pprint(vars(args))  # print出所有超参数

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch  #这里就是为了方便复现模型，这也是为什么在跑实验时候有时候会跑出一样的结果？因为所谓的“随机”数已经“固定”方式产生
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True  # only applies to CUDA convolution operations, and ……（复现结果用？）
        torch.backends.cudnn.benchmark = False  ##内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 一般原则
    # 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = true可以增加运行效率；
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    ### pre- train##
    # ###for debug##
    # args.pre_max_epoch = 1
    # args.MTL = False
    # ###for debug
    need_Pre=1
    if need_Pre==1:
        args.phase='pre_train'
        trainer = PreTrainer(args)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
    #meta+train /MAML
    #现在加入原始 pretrain + ML-test 测试
    need_meta=1
    if need_meta==1:
        i =0
        args.meta_label = args.pre_train_label + i
        args.shot = 10 # pre train 用于 meta val阶段，设置大一点有利于选出泛化能力强一点的pretrain模型？
        print("-------args.shot===",args.shot)
        args.phase='meta_train'
        trainer = MetaTrainer(args)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
    # # 测meta-learning 先不用
    TestSubject=args.TestSubject#传入
    # val 阶段集中在下面
    needPreVal=1
    if needPreVal==True:
        args.phase = 'pre_train'
        print('-----------------pre-val-----------------------')
        originaltest=TestModel(args)
        for i in TestSubject:
            args.TestSubject = [i]
            originaltest.test()

        for i in TestSubject:
            args.TestSubject = [i]
            originaltest.meta_test()
        del originaltest
        torch.cuda.empty_cache()

    print('-----------------meta-val-----------------------')
    args.TestSubject=TestSubject#传回去
    args.phase='meta_eval'

    trainer = MetaTrainer(args)
    for i in TestSubject:
        args.TestSubject = [i]
        trainer.eval()#两次eval挑个好结果
        trainer.eval()

    import sys
    print(sys.getsizeof(trainer) / 1024 / 1024, 'MB')
    end = time.time()  # 算程序运行时间
    print(' using time :%.2f s'%(end-start))