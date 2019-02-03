import sys

sys.path.append('../../../')
from utils import data_iterator
from utils.data_utils import *
import utils.model_pipeline as model_pipeline
from utils.runner import run_stage2
from utils.fetch_global_args import stage2_global_argparser,ALL_CHROMOSOMES
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model_zoo.stage1.basset_factorized_ct import Net as BassetNet
import pdb

# loads pretrained basset_vanilla_ct and fixes its weights
# then learns weights for gene expression (implicitly has access to mean across train cell types)

"""
Usage: python basset+_factorized_nomean_ct.py

Loads pretrained resnet basset weights, specified in specified in @a args.basset_pretrained_path, 
freezing those weights, then learns weights for gene expression. Does NOT have access to the mean!

Uses 2 simple layers at the end to concat genexp with conv_out.

"""


def getargs():
    parser = stage2_global_argparser()
    args = parser.parse_args()
    args = dotdict(vars(args))
    return args



class Net(nn.Module):
    def __init__(self,BASSET_NUM_CELL_TYPES, basset_model, args):
        super(Net, self).__init__()
                
        self.basset_model = basset_model
        # freezing basset_model parameters and setting to eval mode
        if args.freeze_pretrained_model:
            for param in self.basset_model.parameters():
                param.requires_grad = False
            self.basset_model.eval()

        self.layer1 = nn.Linear(4200 + 1630, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.layer3 = nn.Linear(1000,2)
        self.args = args



    def forward(self, s, g):
        if self.args.freeze_pretrained_model: self.basset_model.eval()
        _, conv_out = self.basset_model(s)  # batch_size x BASSET_NUM_CELL_TYPES, batch_size x 4200

        conv_gene = torch.cat([conv_out, g], dim = -1) ## batch_size x 4200 + 1630
        out = F.dropout(F.relu(self.bn1(self.layer1(conv_gene))), p = self.args.dropout, training = self.training)

        out = F.dropout(F.relu(self.bn2(self.layer2(out))), p=self.args.dropout, training=self.training)
        out = self.layer3(out)
        return F.log_softmax(out, dim=1)


def instantiate_model(args, chrm_list=ALL_CHROMOSOMES):
    # load basset model
    basset_checkpoint = torch.load(args.basset_pretrained_path + 'model_best.pth.tar')
    basset_args = basset_checkpoint['args']
    basset_model = BassetNet(basset_args)
    basset_model.load_state_dict(basset_checkpoint['state_dict'])
    if args.cuda: basset_model.cuda()
    BASSET_NUM_CELL_TYPES = basset_args.num_total_cell_types - len(basset_args.validation_list) - len(
        basset_args.test_list)

    # init model
    model = Net(BASSET_NUM_CELL_TYPES, basset_model, args)
    if args.resume_from_best:
        model_pipeline.load_checkpoint(model, checkpoint = args.checkpoint)
        print('LOADED WEIGHTS FROM ' + args.checkpoint + 'model_best.pth.tar')

    di = data_iterator.DataIterator(args.dnase, args.rna_quants, hold_out=args.hold_out,
                                    validation_list=args.validation_list, test_list=args.test_list,
                                    balance_classes_train=True, positive_proportion=args.positive_proportion,
                                    return_locus_mean=False, eval_subsample=500, chromosomes=chrm_list)

    return model, di



if __name__ == '__main__':
    args = getargs()
    print(args)
    model, di = instantiate_model(args, args.chromosomes)
    run_stage2(model,di, args, model_pipeline)



