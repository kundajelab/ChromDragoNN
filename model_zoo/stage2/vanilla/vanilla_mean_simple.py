import sys

sys.path.append('../../../')
from utils import data_iterator
from utils.data_utils import *
import utils.model_pipeline_mean as model_pipeline
from utils.fetch_global_args import stage2_global_argparser
from utils.runner import  run_stage2

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model_zoo.stage1.basset_vanilla_ct import Net as BassetNet


# loads pretrained basset_vanilla_ct and fixes its weights
# then learns weights for gene expression (implicitly has access to mean across train cell types)

def getargs():
    parser = stage2_global_argparser()
    args = parser.parse_args()
    args = dotdict(vars(args))
    return args

class Net(nn.Module):
    def __init__(self, BASSET_NUM_CELL_TYPES, basset_model, args):
        super(Net, self).__init__() 
        self.basset_model = basset_model
        # freezing basset_model parameters and setting to eval mode
        if args.freeze_pretrained_model:
            for param in self.basset_model.parameters():
                param.requires_grad = False
            self.basset_model.eval()

        self.fc1 = nn.Linear(1630 + 1000 + 1, 1000)         # TODO: do not hard-code 1630
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000,2)
        self.args = args
        

    def forward(self, s, g, m):
        if self.args.freeze_pretrained_model: self.basset_model.eval()
        
        _, conv_out = self.basset_model(s)                                                    # batch_size x 1000
        
        g = torch.cat((conv_out,g,m.view(-1,1)), 1)                                           # batch_size x (1000 + 1630 + 1)

        g = F.dropout(F.relu(self.bn1(self.fc1(g))), p=args.dropout, training=self.training)  # batch_size x 1000
        g = F.dropout(F.relu(self.bn2(self.fc2(g))), p=args.dropout, training=self.training)  # batch_size x 1000

        g = self.fc3(g)
        return F.log_softmax(g, dim=1)


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
                                    balance_classes_train=True, positive_proportion = args.positive_proportion,
                                    return_locus_mean=True, eval_subsample=500, chromosomes=chrm_list)


    return model, di
    
    
if __name__ == '__main__':
    args = getargs()
    print(args)
    model, di = instantiate_model(args, args.chromosomes)
    run_stage2(model,di, args, model_pipeline)
