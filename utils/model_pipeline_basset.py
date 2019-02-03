import argparse
import os
import shutil
import time
import random
from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics



def basset_loss(outputs, targets):
    # outputs are softmax values for each cell type, shape batch_size x NUM_CELL_TYPES
    return -torch.sum(targets*F.logsigmoid(outputs) + (1-targets)*(-outputs + F.logsigmoid(outputs)))/targets.size()[0]

def eval(outputs, targets, args, thres=0.5, eps=1e-9):
    return torch.mean((outputs-targets)**2)


def train(model, optimizer, epoch, di, args, criterion=nn.NLLLoss()):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rmses = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    while batch_idx < args.batches_per_epoch:
        seq_batch, target_batch = di.sample_train_batch_basset(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        targets = torch.from_numpy(target_batch).float()

        # measure data loading time
        data_time.update(time.time() - end)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seq_batch, targets = seq_batch.to(device), targets.to(device)
        #if args.cuda:
        #    seq_batch, targets = seq_batch.contiguous().cuda(), targets.contiguous().cuda(async=True)
        #seq_batch, targets = Variable(seq_batch), Variable(targets)

        # compute output
        outputs, _ = model(seq_batch)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        rmse = eval(outputs.data, targets.data, args)

        if np.isnan(loss.data.item()):
            print(outputs.data)
            print('-'*100)
            print(targets.data)
            continue

        rmses.update(rmse, seq_batch.size(0))
        losses.update(loss.data.item(), seq_batch.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | rmse: {rmse:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_epoch,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    rmse=rmses.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, rmse)


def test(model, optimizer, epoch, di, args, criterion=nn.NLLLoss()):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rmses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=args.batches_per_test_epoch)
    batch_idx = 0
    all_preds = []
    all_targets = []

    while batch_idx < args.batches_per_test_epoch:
        # measure data loading time
        data_time.update(time.time() - end)
        seq_batch, target_batch= di.sample_validation_batch_basset(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        targets = torch.from_numpy(target_batch).float()
        if args.cuda:
            seq_batch, targets = seq_batch.contiguous().cuda(), targets.contiguous().cuda() 

        # compute output
        with torch.no_grad():
            outputs, _ = model(seq_batch)
            loss = criterion(outputs, targets)

        # concat to all_preds, all_targets
        index = torch.LongTensor([1])
        if args.cuda:
            index = index.cuda()

        # measure accuracy and record loss
        rmse = eval(outputs.data, targets.data, args)
        rmses.update(rmse, seq_batch.size(0))
        losses.update(loss.data.item(), seq_batch.size(0))

        if batch_idx < args.batches_per_test_epoch - 1:
            all_preds.append(outputs.cpu().data.numpy())
            all_targets.append(targets.cpu().data.numpy())
            try:
                auprc = np.mean([sklearn.metrics.average_precision_score(all_targets[-1][:,i], all_preds[-1][:,i]) for i in range(all_preds[-1].shape[1])])
                auc = np.mean([sklearn.metrics.roc_auc_score(all_targets[-1][:,i], all_preds[-1][:,i]) for i in range(all_preds[-1].shape[1])])
            except:
                auprc, auc = -1, -1
        else:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            auprc = np.mean([sklearn.metrics.average_precision_score(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])
            auc = np.mean([sklearn.metrics.roc_auc_score(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1
        # plot progress
        bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | rmse: {rmse:.3f} | auprc: {auprc:.3f} | auc: {auc:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_test_epoch,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    rmse=rmses.avg,
                    auprc=auprc,
                    auc=auc,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, auprc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def load_checkpoint(model, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(checkpoint, 'model_best.pth.tar')
    if not os.path.exists(filepath):
        raise("No best model in path {}".format(checkpoint))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint
    
def adjust_learning_rate(optimizer, epoch, args):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
