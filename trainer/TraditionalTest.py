# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" TestModel for normal-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
# from dataloader.dataset_loader_BCI_IV_c import DatasetLoader_BCI_IV_subjects as Dataset
# from dataloader.DataSetLoader_BNCI2015004 import DataSetLoader_BNCI2015004 as Dataset
import time
from torch.autograd import Variable# for original test
class TestModel(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args):

        # Set args to be shareable in the class #  #
        self.args = args

        #load dataset
        if args.dataset=='BNCI2015004':
            from dataloader.DataSetLoader_BNCI2015004 import DataSetLoader_BNCI2015004 as Dataset
        elif args.dataset == 'BNCI2014001':
            from dataloader.DataSetLoader_BNCI2014001 import DataSetLoader_BNCI2014001 as Dataset
        elif args.dataset == 'Schirrmeister2017':
            from dataloader.DataSetLoader_Schirrmeister2017 import DataSetLoader_Schirrmeister2017 as Dataset
        elif args.dataset == 'BNCI2014001_SPD':
            from dataloader.DataSetLoader_BNCI2014001_SPD import DataSetLoader_BNCI2014001_SPD as Dataset
        elif args.dataset == 'Schirrmeister2017_SPD':
            from dataloader.DataSetLoader_Schirrmeister2017_SPD import DataSetLoader_Schirrmeister2017_SPD as Dataset
        elif args.dataset == 'BNCI2015004_SPD':
            from dataloader.DataSetLoader_BNCI2015004_SPD import DataSetLoader_BNCI2015004_SPD as Dataset
        else:
            assert print('wrong dataset input')
        print("Preparing dataset loader")
        # Load normal-test set
        self.testset = Dataset('test', self.args, train_aug=False, TrainSubjects=self.args.TrainSubjects, TestSubject=self.args.TestSubject,BinaryClassify = args.BinaryClassify)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # Set pretrain class number
        num_class_pretrain = self.testset.num_class
        in_chans=self.testset.in_chans
        input_time_length=self.testset.time_step
        # Build test model #这
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain,in_chans=in_chans,input_time_length=input_time_length)

        # load pretrained model without classifier block #
        self.model_dict = self.model.state_dict()#
        if self.args.init_weights is not None:#
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            # Set the folder to save the records and checkpoints #
            log_base_dir = './logs/'
            if not osp.exists(log_base_dir):
                os.mkdir(log_base_dir)
            pre_base_dir = osp.join(log_base_dir, 'pre')
            if not osp.exists(pre_base_dir):
                os.mkdir(pre_base_dir)
            save_path1 = '_'.join([args.dataset, args.model_type])
            save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
                args.pre_gamma) + '_step' + \
                         str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            save_path3 = 'TrainSubjects'
            for subject in args.TrainSubjects:
                save_path3 += str(subject)
            save_path3 += '_TestSubject'#这里改了
            for subject in args.TestSubject:
                save_path3 += str(subject)#
            if args.BinaryClassify == 1:
                save_path3 += '_Binary'
            args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2 + '_' + save_path3 + '_' + str(args.pre_train_label)  # 和加上实验日期
            pretrained_dict = torch.load(osp.join(args.save_path, 'max_acc.pth'))['params']#
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}#
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}#
        self.model_dict.update(pretrained_dict)#
        #load the classifier block
        pretrained_dict1 = torch.load(osp.join(args.save_path, 'classifier_max_acc.pth'))['params']
        pretrained_dict1 = {'classifier.' + k: v for k, v in pretrained_dict1.items()}#
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in self.model_dict}#
        self.model_dict.update(pretrained_dict1)#
        self.model.load_state_dict(self.model_dict)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def test(self):
        """The function for the meta-eval phase."""

        # Set model to eval mode # 同orginal_val 一样，只不过换些数据集
        self.model.eval()
        # Load normal-test set#
        #load dataset
        if self.args.dataset=='BNCI2015004':
            from dataloader.DataSetLoader_BNCI2015004 import DataSetLoader_BNCI2015004 as Dataset
        elif self.args.dataset == 'BNCI2014001':
            from dataloader.DataSetLoader_BNCI2014001 import DataSetLoader_BNCI2014001 as Dataset
        elif self.args.dataset == 'Schirrmeister2017':
            from dataloader.DataSetLoader_Schirrmeister2017 import DataSetLoader_Schirrmeister2017 as Dataset
        elif self.args.dataset == 'BNCI2014001_SPD':
            from dataloader.DataSetLoader_BNCI2014001_SPD import DataSetLoader_BNCI2014001_SPD as Dataset
        elif self.args.dataset == 'Schirrmeister2017_SPD':
            from dataloader.DataSetLoader_Schirrmeister2017_SPD import DataSetLoader_Schirrmeister2017_SPD as Dataset
        elif self.args.dataset == 'BNCI2015004_SPD':
            from dataloader.DataSetLoader_BNCI2015004_SPD import DataSetLoader_BNCI2015004_SPD as Dataset
            assert print('wrong dataset input')
        self.testset = Dataset('test', self.args, train_aug=False, TrainSubjects=self.args.TrainSubjects, TestSubject=self.args.TestSubject,BinaryClassify = self.args.BinaryClassify)
        _, valid_results, loss = self.val_orig(self.testset.X_test,self.testset.y_test)  # p
        print('-------OriginalTest for Pre-train phase----------------------------------------------------')
        print('test subject:',self.args.TestSubject[0])
        print('OriginalTest--Test accuracy-(ACC):', valid_results[0], 'F-mearsure:', valid_results[2],"loss:", loss.item())  #
        print('ACC:', valid_results[0])
        print('F-mearsure:', valid_results[2])
    def val_orig(self, X_val, y_val):  # ML-validation
        predicted_loss = []
        inputs = torch.from_numpy(X_val)
        labels = torch.FloatTensor(y_val * 1.0)
        inputs, labels = Variable(inputs), Variable(labels)

        results = []
        predicted = []

        self.model.eval()
        self.model.mode = 'origval'

        if torch.cuda.is_available():
            inputs = inputs.type(torch.cuda.FloatTensor)
        else:
            inputs = inputs.type(torch.FloatTensor)

        predicted = self.model(inputs)
        labels1 = torch.IntTensor(y_val * 1.0)
        labels1 = labels1.type(torch.int64)  #
        labels1 = labels1.cuda()
        loss = F.cross_entropy(predicted, labels1)
        predicted = predicted.data.cpu().numpy()

        Y = labels.data.numpy()
        predicted = np.argmax(predicted, axis=1)

        for param in ["acc", "recall", "precision","fmeasure", "auc"]:
            if param == 'acc':
                results.append(accuracy_score(Y, np.round(predicted)))
            if param == "recall":
                results.append(recall_score(Y, np.round(predicted), average='micro'))
            if param == "fmeasure":
                precision = precision_score(Y, np.round(predicted), average='micro')
                recall = recall_score(Y, np.round(predicted), average='micro')
                results.append(2*precision*recall/ (precision+recall))
            if param == "auc":
                # Y = label.data.cpu().numpy()#
                def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
                    lb = LabelBinarizer()
                    lb.fit(y_test)
                    y_test = lb.transform(y_test)
                    y_pred = lb.transform(y_pred)
                    return roc_auc_score(y_test, y_pred, average=average)
                auc = multiclass_roc_auc_score(Y, predicted)  ##
                results.append(auc)
        return predicted, results, loss

    def meta_test(self):#For meta-test to the Pre-train model
        """The function for the meta-eval phase."""
        # Load the logs
        def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_test = lb.transform(y_test)
            y_pred = lb.transform(y_pred)
            return roc_auc_score(y_test, y_pred, average=average)

        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        # Load meta-test set#TODO: 也许可以更改数据集的方式，比如 ”train-meta" 作为输入等
        args = self.args
        if args.dataset == 'BNCI2015004':
            from dataloader.DataSetLoader_BNCI2015004 import DataSetLoader_BNCI2015004 as Dataset
        elif args.dataset == 'BNCI2014001':
            from dataloader.DataSetLoader_BNCI2014001 import DataSetLoader_BNCI2014001 as Dataset
        elif args.dataset == 'Schirrmeister2017':
            from dataloader.DataSetLoader_Schirrmeister2017 import DataSetLoader_Schirrmeister2017 as Dataset
        elif args.dataset == 'BNCI2014001_SPD':
            from dataloader.DataSetLoader_BNCI2014001_SPD import DataSetLoader_BNCI2014001_SPD as Dataset
        elif args.dataset == 'Schirrmeister2017_SPD':
            from dataloader.DataSetLoader_Schirrmeister2017_SPD import DataSetLoader_Schirrmeister2017_SPD as Dataset
        elif args.dataset == 'BNCI2015004_SPD':
            from dataloader.DataSetLoader_BNCI2015004_SPD import DataSetLoader_BNCI2015004_SPD as Dataset
        else:
            assert print('wrong dataset input')
        print('---------meta-test for Pre-train phase --------------------------------------------------')
        print("Preparing meta-valdataset loader")
        print('test subject:', self.args.TestSubject[0])
        test_set = Dataset('test', self.args, TrainSubjects=self.args.TrainSubjects, ValSubject=self.args.ValSubject,
                           TestSubject=self.args.TestSubject, BinaryClassify=args.BinaryClassify)
        sampler = CategoriesSampler(test_set.label, 20, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((20,))  #
        test_f1_record = np.zeros((20,))
        test_auc_record = np.zeros((20,))
        # Load model for meta-test phase

        # Set model to eval mode
        self.model.eval()
        self.model.mode = 'preval'
        # Set accuracy averager
        ave_acc = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        Y = label.data.cpu().numpy()
        # Start meta-test
        for i, batch in enumerate(loader, 1):  ##
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            logits = self.model((data_shot, label_shot, data_query))
            acc = count_acc(logits, label)
            logits = logits.data.cpu().numpy()  ##
            predicted = np.argmax(logits, axis=1)
            f1 = f1_score(Y, predicted, average='macro')
            auc = multiclass_roc_auc_score(Y, predicted)  ##
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            test_f1_record[i - 1] = f1  #
            test_auc_record[i - 1] = auc
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        f1_m, f1_pm = compute_confidence_interval(test_f1_record)
        auc_m, auc_pm = compute_confidence_interval(test_auc_record)

        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'],
                                                                      ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print('Test f1 {:.4f} + {:.4f}'.format(f1_m, f1_pm))
        print('Test auc {:.4f} + {:.4f}'.format(auc_m, auc_pm))