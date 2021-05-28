import argparse
import os
from datetime import datetime
import random

import torch
import numpy as np

import sklearn.metrics as metrics
from dotmap import DotMap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import optim, nn
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset.ChexpertDataloader import data_loader_dict
from environment_setup import PROJECT_ROOT_DIR, read_config
from loss_fn.WeightedBCE import WeightedBCEWithLogitLoss, WeightedBCEWithLogitLossSec
from models.model_factory import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CheXpertTrainer():

    @staticmethod
    def scale_lr(lr, factor, optimizer):
        """
        Scale the learning rate of the optimizer
        :return: in-place update of parameters
        """
        lr = lr * factor
        print(f"Updating the learning rate to {lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    @staticmethod
    def epochTrain(model, dataLoader_train, dataLoaderVal, optimizer, criterion, logger_train, logger_val, epochId,
                   max_val_auc, scaler, args):

        model.train()
        for batchID, (varInput, target) in enumerate(dataLoader_train):

            optimizer.zero_grad()
            varTarget = target.cuda(non_blocking=True)
            varInput = varInput.to(DEVICE)
            # Runs the forward pass with autocasting.
            with autocast():
                varOutput = model(varInput)
                lossvalue = criterion(varOutput, varTarget)

            scaler.scale(lossvalue).backward()
            # optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            l = lossvalue.item()

            if batchID % 1000 == 999:
                step = (epochId * len(dataLoader_train)) + batchID  # Use the same step for logging
                logger_train.add_scalar(tag="loss", scalar_value=l, global_step=step)
                le, val_auc = CheXpertTrainer.epochVal(model=model, dataLoader=dataLoaderVal, loss=criterion, args=args)
                logger_val.add_scalar(tag="loss", scalar_value=le.item(), global_step=step)
                logger_val.add_scalar(tag="mean_auc", scalar_value=val_auc, global_step=step)
                # Put the model back in training mode
                model.train()
                if val_auc > max_val_auc:
                    torch.save({'epoch': epochId + 1, 'state_dict': model.state_dict(), 'best_auc': max_val_auc,
                                'optimizer': optimizer.state_dict()},
                               os.path.join(args.save_dir, 'model' + str(epochId) + '.pth'))
                print(f"Epoch {epochId}:{batchID} iterations completed")

        # lr scheduling
        if args.lr_schedule and (epochId + 1) % args.schedule_after_epochs == 0:
            args.lr = CheXpertTrainer.scale_lr(lr=args.lr, factor=args.factor, optimizer=optimizer)

        return max_val_auc

    @staticmethod
    def epochVal(model, dataLoader, loss, args):
        model.eval()
        lossVal = 0
        lossValNorm = 0

        # Code for computing AUC
        outGT = torch.FloatTensor().to(DEVICE)
        outPRED = torch.FloatTensor().to(DEVICE)

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoader):
                varInput = varInput.to(DEVICE)
                target = target.cuda(non_blocking=True)
                outGT = torch.cat((outGT, target), 0).to(DEVICE)

                varOutput = model(varInput)

                losstensor = loss(varOutput, target)
                lossVal += losstensor
                lossValNorm += 1

                outPRED = torch.cat((outPRED, varOutput), 0)

        outLoss = lossVal / lossValNorm
        # Also compute the AUC metric
        aurocIndividual = CheXpertTrainer.computeAUROC(dataGT=outGT, dataPRED=outPRED, classCount=args.class_count)
        aurocMean = np.array(aurocIndividual).mean()
        return outLoss, aurocMean

    @staticmethod
    def perform_train_val(model, dataLoaderTrain, dataLoaderVal, trMaxEpoch, logdir, checkpoint, label_weights, lr,
                          args):

        # SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        if label_weights is not None:
            pos_weight, neg_weight = label_weights
            pos_weight = pos_weight.to(DEVICE)
            neg_weight = neg_weight.to(DEVICE)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = WeightedBCEWithLogitLossSec(pos_weight=pos_weight, neg_weight=neg_weight)
        # LOAD CHECKPOINT
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        logger_train = SummaryWriter(os.path.join(logdir, 'train'))
        logger_val = SummaryWriter(os.path.join(logdir, 'val'))

        # TRAIN THE NETWORK
        max_val_auc = 0

        # Mixed Precision Training
        scaler = GradScaler()
        for epochID in range(trMaxEpoch):
            # (model, dataLoader_train, dataLoaderVal, optimizer, criterion, logger_train, logger_val, epochId
            max_val_auc = CheXpertTrainer.epochTrain(model=model, dataLoader_train=dataLoaderTrain,
                                                                 dataLoaderVal=dataLoaderVal,
                                                                 optimizer=optimizer, criterion=criterion,
                                                                 logger_train=logger_train,
                                                                 logger_val=logger_val, epochId=epochID,
                                                                 max_val_auc=max_val_auc, scaler=scaler,
                                                                 args=args)
        # Select the last saved model as it would work as the best model
        bestModel = max([int(''.join(i for i in x if i.isdigit())) for x in os.listdir(args.save_dir)])
        return bestModel

    @staticmethod
    def computeAUROC(dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC

    @staticmethod
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):

        cudnn.benchmark = True

        if checkpoint != None and torch.cuda.is_available():
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        outGT = torch.FloatTensor().to(DEVICE)
        outPRED = torch.FloatTensor().to(DEVICE)

        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                input = input.to(DEVICE)
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).to(DEVICE)

                out = model(input)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        print('AUROC mean ', aurocMean)

        for i in range(0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])

        return outGT, outPRED

    def run(self, args):
        # Class names
        class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        nn_class_count = len(class_names)
        data_loader, label_weights = data_loader_dict(uncertainty_labels=args.uncertainty_labels,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers, build_grph=args.build_graph)
        model = create_model(args.model_type).to(DEVICE)
        args.class_count = nn_class_count
        if args.mode == 'train':
            bestModelNumber = self.perform_train_val(model=model, dataLoaderTrain=data_loader['train'],
                                                     dataLoaderVal=data_loader['valid'], trMaxEpoch=args.max_epoch,
                                                     logdir=args.logdir, checkpoint=args.pretrained_checkpoint,
                                                     label_weights=label_weights, lr=args.lr, args=args)
            print("Model trained")
        else:
            bestModelNumber = args.model_number
            print(f"Test mode with model {bestModelNumber}")
        checkpoint = os.path.join(args.save_dir, 'model' + str(bestModelNumber) + '.pth')
        self.plot_results(model=model, dataLoaderTest=data_loader['test'], nnClassCount=nn_class_count,
                          checkpoint=checkpoint)

    def plot_results(self, model, dataLoaderTest, nnClassCount, checkpoint):
        class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        outGT1, outPRED1 = CheXpertTrainer.test(model=model, dataLoaderTest=dataLoaderTest, nnClassCount=nnClassCount,
                                                checkpoint=checkpoint, class_names=class_names)

        # outGT3, outPRED3 = CheXpertTrainer.test(model=model, dataLoaderTest=dataLoaderTest, nnClassCount=nnClassCount,
        #                                         checkpoint="model_zeros.pth.tar", class_names=class_names)

        for i in range(nnClassCount):
            fpr, tpr, threshold = metrics.roc_curve(outGT1.cpu()[:, i], outPRED1.cpu()[:, i])
            roc_auc = metrics.auc(fpr, tpr)
            f = plt.subplot(2, 7, i + 1)
            # fpr2, tpr2, threshold2 = metrics.roc_curve(outGT3.cpu()[:,i], outPRED3.cpu()[:,i])
            # roc_auc2 = metrics.auc(fpr2, tpr2)

            plt.title('ROC for: ' + class_names[i])
            plt.plot(fpr, tpr, label='U-ones: AUC = %0.2f' % roc_auc)
            # plt.plot(fpr2, tpr2, label='U-zeros: AUC = %0.2f' % roc_auc2)

            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 30
        fig_size[1] = 10
        plt.rcParams["figure.figsize"] = fig_size

        plt.savefig("ROC1345.png", dpi=1000)
        plt.show()


def set_seeds():
    random.seed(42)
    np.random.seed(42)


if __name__ == '__main__':
    set_seeds()
    args = DotMap()
    parser = read_config()
    args.max_epoch = parser['setup'].getint('max_epoch')
    args.batch_size = parser['setup'].getint('batch_size')
    args.num_workers = parser['setup'].getint('num_workers')
    args.model_type = parser['setup'].get('model_type')
    args.logdir = parser['setup'].get('log_dir')
    args.save_dir = parser['setup'].get('save_dir')
    args.lr = parser['setup'].getfloat('lr')
    args.mode = parser['setup'].get('mode')
    args.model_number = parser['setup'].getint('model_number')
    args.lr_schedule = parser['setup'].getboolean('lr_schedule')
    args.factor = parser['setup'].getfloat('factor')
    args.schedule_after_epochs = parser['setup'].getint('schedule_after_epochs')

    args.uncertainty_labels = parser['data'].get('uncertainty_labels')
    args.build_graph = True #args.model_type == 'base'
    # Add some extra configurations
    args.pathFileTrain = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'CheXpert-v1.0-small', 'train.csv')
    args.pathFileValid = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'CheXpert-v1.0-small', 'valid.csv')
    args.logdir = os.path.join(PROJECT_ROOT_DIR, 'results', args.logdir, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    args.save_dir = os.path.join(PROJECT_ROOT_DIR, 'results', args.save_dir, args.model_type)
    if not os.path.exists(args.logdir) or not os.path.exists(args.save_dir):
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
    if parser['setup'].getboolean('continue_training'):
        args.pretrained_checkpoint = os.path.join(args.save_dir,
                                                  'model' + parser['setup'].get('continue_train_model_number') + '.pth')
    else:
        args.pretrained_checkpoint = None
    CheXpertTrainer().run(args=args)
