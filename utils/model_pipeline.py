import argparse
import os
import shutil
import time
import random
from multiprocessing import  Pool
from pytorch_classification.utils import Bar, Logger, AverageMeter
import numpy as np
import math
from utils.data_utils import *
import joblib
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics

"""
Example use (roughly):
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('/home/katherin/CS273B/')
from utils import data_iterator
from utils.data_utils import *
import model_pipeline as pipeline

di = data_iterator.DataIterator(
   '/mnt/data/packbited/',
    '/mnt/data/rna_quants_1630tf.joblib', hold_out='chromosomes', validation_list=[1,2], test_list=[3])

args = dotdict({
    'lr': 0.01,
    'momentum': 0.5,
    'epochs': 5,
    'batch_size': 256,
    'batches_per_epoch': 1000,
    'test_batches_per_epoch': 10,
    'log_interval': 500,
    'cuda': True,
})

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1000, 10, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x),2))
        x = x.view(-1, 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
        
        
model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
start_epoch = 0
best_acc = 0

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    train_loss, train_acc = pipeline.train(model, optimizer, epoch, di, args)
    test_loss, test_acc = pipeline.test(model, optimizer, epoch, di, args)

    # append logger file
    logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    pipeline.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

"""

def eval(outputs, targets, args, thres=0.5, eps=1e-9):
    index = torch.LongTensor([1])
    if args.cuda:
        index = index.cuda()
    outputs_class = torch.index_select(outputs, 1, index=index).view(-1)
    outputs_class = torch.exp(outputs_class)
    preds = torch.ge(outputs_class.float(), thres).float()
    targets = targets.float()
    true_positive = (preds * targets).sum()
    precis = true_positive / (preds.sum() + eps)
    recall = true_positive / (targets.sum() + eps)
    f1 = 2 * precis * recall / (precis + recall + eps)
    return (precis, recall, f1)


def train(model, optimizer, epoch, di, args, criterion=nn.NLLLoss()):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precis = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    all_preds = np.array([])
    all_targets = np.array([])

    while batch_idx < args.batches_per_epoch:
        seq_batch, gene_batch, target_batch = di.sample_train_batch(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)
        targets = torch.from_numpy(target_batch)

        # measure data loading time
        data_time.update(time.time() - end)

        # predict
        if args.cuda:
            seq_batch, gene_batch, targets = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda(), targets.cuda(async=True)
        seq_batch, gene_batch, targets = Variable(seq_batch), Variable(gene_batch), Variable(targets)

        # compute output
        outputs = model(seq_batch, gene_batch)
        loss = criterion(outputs, targets)

        # concat to all_preds, all_targets
        index = Variable(torch.LongTensor([1]))
        if args.cuda:
            index = index.cuda()
        all_preds = np.concatenate((all_preds, torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy()))
        all_targets = np.concatenate((all_targets, targets.cpu().data.numpy()))

        # measure accuracy and record loss
        p, r, f = eval(outputs.data, targets.data, args)
        precis.update(p, seq_batch.size(0))
        recall.update(r, seq_batch.size(0))
        f1.update(f, seq_batch.size(0))
        losses.update(loss.item(), seq_batch.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | prec: {precis:.3f} | rec: {recall:.3f} | f1: {f1:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_epoch,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    precis=precis.avg,
                    recall=recall.avg,
                    f1=f1.avg,
                    )
        bar.next()
    bar.finish()

    # compute train auprc/auc for direct comparison to test
    train_auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
    train_auc = sklearn.metrics.roc_auc_score(all_targets, all_preds)
    print('train auprc: {auprc: .3f} | train auc: {auc: .3f}'.format(
        auprc=train_auprc,
        auc=train_auc,
    ))

    return (losses.avg, f1.avg)


def test(model, optimizer, epoch, di, args, criterion=nn.NLLLoss()):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precis = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=args.batches_per_test_epoch)
    batch_idx = 0
    all_preds = np.array([])
    all_targets = np.array([])

    while batch_idx < args.batches_per_test_epoch:
        # measure data loading time
        data_time.update(time.time() - end)
        seq_batch, gene_batch, target_batch= di.sample_validation_batch(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)
        targets = torch.from_numpy(target_batch)
        if args.cuda:
            seq_batch, gene_batch, targets = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda(), targets.cuda()
        seq_batch, gene_batch, targets = Variable(seq_batch, volatile=True), Variable(gene_batch, volatile=True), Variable(targets)

        # compute output
        outputs = model(seq_batch, gene_batch)
        loss = criterion(outputs, targets)

        # concat to all_preds, all_targets
        index = Variable(torch.LongTensor([1]))
        if args.cuda:
            index = index.cuda()
        all_preds = np.concatenate((all_preds, torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy()))
        all_targets = np.concatenate((all_targets, targets.cpu().data.numpy()))

        # measure accuracy and record loss
        p, r, f = eval(outputs.data, targets.data, args)
        auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
        auc = sklearn.metrics.roc_auc_score(all_targets, all_preds)
        precis.update(p, seq_batch.size(0))
        recall.update(r, seq_batch.size(0))
        f1.update(f, seq_batch.size(0))
        losses.update(loss.item(), seq_batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1
        # plot progress
        bar.suffix  = '({batch}/{size}) | Loss: {loss:.4f} | precis: {precis:.3f} | recall: {recall:.3f} | f1: {f1:.3f} | auprc: {auprc:.3f} | auc: {auc:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_test_epoch,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    precis=precis.avg,
                    recall=recall.avg,
                    f1=f1.avg,
                    auprc=auprc,
                    auc=auc,
                    )
        bar.next()
    bar.finish()

    val_results = {'preds': all_preds, 'labels': all_targets}
    joblib.dump(val_results, os.path.join(args.checkpoint, 'validation_results.joblib'))

    return (losses.avg, auprc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
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

def adjust_learning_rate(optimizer, epoch, args, state):
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



def evaluate_model(model, args, di, labels_avail=True, type='test', mode='report'):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if type=='test': num_examples, chrms, cell_types = di.num_test_examples, di.test_chrms, di.test_cell_types
    elif type=='validation': num_examples, chrms, cell_types = di.num_validation_examples, di.validation_chrms, di.validation_cell_types
    else: raise Exception("type is one of [train, validation, test]")

    end = time.time()
    max_batches = int(math.ceil(num_examples/(di.eval_subsample*args.batch_size)))+(len(chrms)*len(cell_types))
    bar = Bar('Processing', max=max_batches)
    # + |test_chrms|*|test_cell_types| is to account for subsampling starting from each chromosome in the worst case
    batch_idx = 0
    all_preds = []

    for seq_batch, gene_batch in di.eval_generator(args.batch_size, type):
        data_time.update(time.time() - end)
        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)
        if args.cuda:
            seq_batch, gene_batch = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda()
        seq_batch, gene_batch = Variable(seq_batch, volatile=True), Variable(gene_batch, volatile=True)

        # compute output
        outputs = model(seq_batch, gene_batch)
        index = Variable(torch.LongTensor([1]))
        if args.cuda:
            index = index.cuda()
        all_preds.append(torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix  = '({batch}/{size}) | Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                    batch=batch_idx,
                    size=max_batches,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()

    all_preds = np.concatenate(all_preds)

    if mode=='save_preds':
        ctype_chr_pred_dict, *_ = di.populate_ctype_chr_pred_dict(cell_types, chrms, all_preds, ret_labels=labels_avail)       

        # assumes outputs are log-softmaxed, taking exponents
        for ctype in ctype_chr_pred_dict:
            for chrm in ctype_chr_pred_dict[ctype]:
                ctype_chr_pred_dict[ctype][chrm]['preds'] = np.exp(ctype_chr_pred_dict[ctype][chrm]['preds'])             
 
        print('ALL PREDICTIONS READY, SAVING THEM')
        matrix_preds = flatten_dict_of_dicts(ctype_chr_pred_dict)

        joblib.dump(ctype_chr_pred_dict, os.path.join(args.checkpoint, type + '_preds.joblib'))
        joblib.dump(matrix_preds, os.path.join(args.checkpoint, type + '_matrix_preds.joblib'))
        
        if labels_avail:
            matrix_labels = flatten_dict_of_dicts(ctype_chr_pred_dict, 'labels')
            joblib.dump(matrix_labels, os.path.join(args.checkpoint, type + '_matrix_labels.joblib'))
        
    elif mode=='report':
        print('ALL PREDICTIONS READY, PREPARING PLOTS')
        di.evaluate_model(all_preds, type, args.checkpoint, args.report_filename)

    else: raise Exception("mode is one of [report, save_preds]")

