import sys

sys.path.append('../../')
from utils import data_iterator
from utils.data_utils import *
from utils.fetch_global_args import stage2_global_argparser
import utils.runner as runner

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


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

        self.layer1 = nn.Linear(4200 + args.num_genes, 1000)    # all stage1 models have dim 4200 for conv out
        self.bn1 = nn.BatchNorm1d(1000)

        if args.with_mean:
            self.layer2 = nn.Linear(1000 + BASSET_NUM_CELL_TYPES + 2, 100)
        else:
            self.layer2 = nn.Linear(1000 + BASSET_NUM_CELL_TYPES + 1, 100)

        self.bn2 = nn.BatchNorm1d(100)
        self.layer3 = nn.Linear(100, 2)
        self.args = args

    def forward(self, s, g, m=None):
        if self.args.freeze_pretrained_model: self.basset_model.eval()
        basset_out, conv_out = self.basset_model(s)  # batch_size x BASSET_NUM_CELL_TYPES, batch_size x 4200
        basset_out_mean = torch.mean(torch.sigmoid(basset_out), 1, True)  # batch_size x 1
        conv_gene = torch.cat([conv_out, g], dim = -1) ## batch_size x 4200 + 1630
        out = F.dropout(F.relu(self.bn1(self.layer1(conv_gene))), p = self.args.dropout, training = self.training)  # batch_size

        if self.args.with_mean:
            out = torch.cat([out, basset_out, basset_out_mean, m.view(-1, 1)], dim = -1)
        else:
            out = torch.cat([out, basset_out, basset_out_mean], dim = -1)

        out = F.dropout(F.relu(self.bn2(self.layer2(out))), p=self.args.dropout, training=self.training)
        out = self.layer3(out)
        return F.log_softmax(out, dim=-1)

if __name__ == '__main__':
    args = getargs()
    print(args)
    if args.with_mean:
        import utils.model_pipeline_mean as model_pipeline
    else:
        import utils.model_pipeline as model_pipeline

    model = runner.instantiate_model_stage2(args, Net, model_pipeline)
    di = runner.load_data_iterator_stage2(args)
    runner.run_stage2(model,di, args, model_pipeline)
