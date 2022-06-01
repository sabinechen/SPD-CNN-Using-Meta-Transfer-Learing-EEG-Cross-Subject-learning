##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable# for original validation
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
# from dataloader.dataset_loader_BCI_IV_c import DatasetLoader_BCI_IV_subjects as Dataset
# from dataloader.DataSetLoader_BNCI2015004_New import DataSetLoader_BNCI2015004 as Dataset

class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints #输出文件记录关键要参数
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        # save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
        #     '_maxepoch' + str(args.pre_max_epoch)+\
        #     '_' + str(args.meta_label)
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        #save_path3用于记录哪个subject用于train哪个用于test
        save_path3='TrainSubjects'
        for subject in args.TrainSubjects:
            save_path3+=str(subject)
        save_path3+='_TestSubject'
        for subject in args.TestSubject:
            save_path3+=str(subject)
        # 判断是否为二分类：
        if args.BinaryClassify == 1:
            save_path3 += '_Binary'
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2+'_' + save_path3+'_' + str(args.pre_train_label)#和加上实验日期
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
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
        self.trainset = Dataset('train', self.args, train_aug=False,TrainSubjects=self.args.TrainSubjects,ValSubject=self.args.ValSubject,TestSubject=self.args.TestSubject,BinaryClassify = args.BinaryClassify)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True,drop_last=False)

        # Load meta-val set
        self.valset = Dataset('val', self.args,TrainSubjects=self.args.TrainSubjects,ValSubject=self.args.ValSubject,TestSubject=self.args.TestSubject,BinaryClassify = args.BinaryClassify)# PS:import DataSetLoader_BNCI2015004 as Dataset
        self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)#用20多少个task去验证
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class
        in_chans=self.trainset.in_chans
        input_time_length=self.trainset.time_step
        # Build pretrain model #
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain,in_chans=in_chans,input_time_length=input_time_length)
        #self.model=self.model.float()
        # Set optimizer
        params=list(self.model.encoder.parameters())+list(self.model.classifier.parameters())
        self.optimizer=optim.Adam(params,lr=args.pre_lr)#
        # def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))# save parameter of the FeatureExtractor
        torch.save(dict(params=self.model.classifier.state_dict()), osp.join(self.args.save_path, 'classifier_'+ name + '.pth'))# save parameter of the Classifier block

    def train(self):
        """The function for the pre-train phase."""
        def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_test = lb.transform(y_test)
            y_pred = lb.transform(y_pred)
            return roc_auc_score(y_test, y_pred, average=average)
        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['meta_val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['meta_val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['meta_val_max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        trlog['meta_val_max_acc_epoch'] = 0


        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        
        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode
            #
            print('Epoch {}'.format(epoch))
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
                
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                del data,batch
                # Calculate loss and train accuracy/ train auc
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                del label
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                del loss,acc
            torch.cuda.empty_cache()
            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            
          ###### start the original ML evaluation  #
            self.model.eval()
            self.model.mode='origval'
            #
            _, valid_results,loss=self.val_orig(self.valset.X_val,self.valset.y_val)  #
            print ('OriginalValidation--validation accuracy-(ACC):', valid_results[0],'F-mearsure:',valid_results[2],'auc:',valid_results[3],"loss:",loss.item())
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(loss.item()), epoch)
            writer.add_scalar('data/val_acc', valid_results[0], epoch)
            #
            del loss
            torch.cuda.empty_cache()
            # Update best saved model
            if valid_results[0] > trlog['max_acc']:
                trlog['max_acc'] = valid_results[0]
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')  # 这里save去logs参数里面，保存模型并且命名为"max_acc:

          ### Start meta_validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            meta_val_loss_averager = Averager()
            meta_val_acc_averager = Averager()

            # Generate the labels for test
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


          ### Run meta-validation #
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                    del batch
                else:
                    data = batch[0]
                    del batch
                #data=data.float()
                p = self.args.shot * self.args.way#
                data_shot, data_query = data[:p], data[p:]#
                del data
                logits = self.model((data_shot, label_shot, data_query))
                del data_shot, data_query
                torch.cuda.empty_cache()
                # Calculate loss and train accuracy/ train auc
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                meta_val_loss_averager.add(loss.item())
                meta_val_acc_averager.add(acc)

            # Update validation averagers
            meta_val_loss_averager = meta_val_loss_averager.item()
            meta_val_acc_averager = meta_val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/meta_val_loss', float(meta_val_loss_averager), epoch)
            writer.add_scalar('data/meta_val_acc', float(meta_val_acc_averager), epoch)

            # Update best saved model
            if meta_val_acc_averager > trlog['meta_val_max_acc']:
                trlog['meta_val_max_acc'] = meta_val_acc_averager
                trlog['meta_val_max_acc_epoch'] = epoch
                self.save_model('meta_val_max_acc')  # 这里save去logs参数里面，保存模型并且命名为"max_acc:
                # self.save_model('max_acc')#这里save去哪里呢
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['meta_val_loss'].append(meta_val_loss_averager)
            trlog['meta_val_acc'].append(meta_val_acc_averager)
            # print the current acc during the training
            if self.args.P300==1:
                print('Meta-Validation--currentval_auc_averager',meta_val_acc_averager)
            else:
                print('Meta-Validation--currentval_acc_averager', meta_val_acc_averager)
            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        writer.close()
        #   ### End of  meta_validation for this epoch, set model to eval mode #
        print('Meta-Validation--Max_acc_epoch:', trlog['meta_val_max_acc_epoch'], 'max_val_acc(ACC):', trlog['meta_val_max_acc'])
        print('Original-Validation--Max_acc_epoch:', trlog['max_acc_epoch'], 'max_val_acc(ACC):',trlog['max_acc'])
        print('---------------------------------------END-OF-PRE--------------------------------------------------')
        # Save log
        torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
    def val_orig(self, X_val, y_val):  # ml_validation
        predicted_loss=[]
        inputs = torch.from_numpy(X_val)
        labels = torch.FloatTensor(y_val*1.0)
        inputs, labels = Variable(inputs), Variable(labels)
        
        results = []
        predicted = []
                
        self.model.eval()
        self.model.mode = 'origval'
        

        if torch.cuda.is_available():
            inputs= inputs.type(torch.cuda.FloatTensor)
        else:
            inputs = inputs.type(torch.FloatTensor)

        predicted=self.model(inputs)
        labels1=torch.IntTensor(y_val*1.0)
        labels1=labels1.type(torch.int64)#
        labels1=labels1.cuda() #
        loss = F.cross_entropy(predicted, labels1)
        predicted= predicted.data.cpu().numpy()

        Y=labels.data.numpy()
        predicted=np.argmax(predicted, axis=1)

        del inputs
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
                # Y = label.data.cpu().numpy()#本来就是
                def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
                    lb = LabelBinarizer()
                    lb.fit(y_test)
                    y_test = lb.transform(y_test)
                    y_pred = lb.transform(y_pred)
                    return roc_auc_score(y_test, y_pred, average=average)
                auc = multiclass_roc_auc_score(Y, predicted)  ##
                results.append(auc)

        return predicted, results,loss



