import sys

sys.path.append('../../../')
from utils import data_iterator
from utils.data_utils import *
import utils.model_pipeline_mean as model_pipeline
from utils.fetch_global_args import stage2_global_argparser
import utils.runner as runner

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

    
if __name__ == '__main__':
    args = getargs()
    print(args)
    model = runner.instantiate_model_stage2(args, BassetNet, Net, model_pipeline)
    di = runner.load_data_iterator_stage2(args, return_locus_mean=True)
    runner.run_stage2(model,di, args, model_pipeline)
