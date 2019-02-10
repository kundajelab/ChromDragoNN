import sys
sys.path.append('../../')
from utils import data_iterator
from utils.data_utils import *
import utils.model_pipeline_basset as model_pipeline
import utils.runner as runner
from utils.fetch_global_args import stage1_global_argparser,ALL_CHROMOSOMES
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def getargs():
    parser = stage1_global_argparser()
    args = parser.parse_args()
    args = dotdict(vars(args))

    return args


# training and test will be from all chromosomes and training cell types
# this model will be used later for transfer learning to new cell types


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.dropout = args.dropout
        self.num_cell_types = args.num_total_cell_types - len(args.validation_list) - len(args.test_list)

        self.conv1 = nn.Conv2d(4, 300, (19, 1), stride = (1, 1), padding=(9,0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride = (1, 1), padding = (5,0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride = (1, 1), padding = (4,0))


        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()                          # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)                                   # batch_size x 4 x 1000 x 1 [4 channels]
        s = self.maxpool1(F.relu(self.bn1(self.conv1(s))))           # batch_size x 300 x 333 x 1
        s = self.maxpool2(F.relu(self.bn2(self.conv2(s))))           # batch_size x 200 x 83 x 1
        s = self.maxpool3(F.relu(self.bn3(self.conv3(s))))           # batch_size x 200 x 21 x 1
        s = s.view(-1, 4200)
        conv_out = s

        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        
        
        s = self.fc3(s)

        return s, conv_out


if __name__ == '__main__':
    args = getargs()
    print(args)
 
    model = runner.instantiate_model_stage1(args, Net, model_pipeline)
    di = runner.load_data_iterator_stage1(args)
    runner.run_stage1(model, di, args, model_pipeline)
