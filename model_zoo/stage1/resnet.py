import sys
import os
sys.path.append('../../')
from utils import data_iterator
from utils.data_utils import *
import utils.model_pipeline_basset as model_pipeline
import utils.runner as runner
from utils.fetch_global_args import stage1_global_argparser,ALL_CHROMOSOMES

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


"""
Usage: python basset_resnet_one.py

Notes: This Model implements a residual convolutional network for traditional Basset:
It predicts chromatin accessibility at a given loci accross the training cell-types,
returning the predictions as well as the final convolutional layer

- All conv layers are followed by batch normalization and relu activation
The network is broken down into four residual blocks:
L1Block --> 2 convolutional layers, 64 channels, filter size (3,1)
L2Block --> 2 convolutional layers, 128 channels, filter size (7,1)
L3Block --> 3 convolutional layers, 200 channels, filter size (7,1), (3,1),(3,1)
L4Block --> 2 convolutional layers, 200 channels, filter size (7,1)


The number of each residual block is specified as a constructor argument to net:
Ex) Net([2,2,2,2])

The complete network consists of:
- 2 convolutional layers --> 48, 64 channels, respecitively, filter size (3,1)
- 2 x L1Block
- 1 conv layer
- 2 x L2Block
- maxpool
- 1 conv layer
- 2 x L3Block
- maxpool
- 2 x L4Block
- 1 conv layer
- 2 fully connected layers



Splits training and testing set as specified in @a args, saving checkpoints specified by args.
Specify learning rates, batch_sizes, etc...


"""

def getargs():
    parser = stage1_global_argparser()
    parser.add_argument('-rbl', '--blocks', nargs = 4, type=int, default=[2, 2, 2, 2])
    args = parser.parse_args()
    args = dotdict(vars(args))

    return args


# training and test will be from all chromosomes and training cell types
# this model will be used later for transfer learning to new cell types


class L1Block(nn.Module):

    def __init__(self):
        super(L1Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L2Block(nn.Module):

    def __init__(self):
        super(L2Block, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L3Block(nn.Module):

    def __init__(self):
        super(L3Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(200)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)

        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2, nn.ReLU(inplace=True),
                                   self.conv3, self.bn3)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L4Block(nn.Module):

    def __init__(self):
        super(L4Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn2 = nn.BatchNorm2d(200)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.dropout = args.dropout
        self.num_cell_types = args.num_total_cell_types - len(args.validation_list) - len(args.test_list)

        self.conv1 = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.prelayer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                      self.conv2, self.bn2, nn.ReLU(inplace=True))


        self.layer1 = nn.Sequential(*[L1Block() for x in range(args.blocks[0])])
        self.layer2 = nn.Sequential(*[L2Block() for x in range(args.blocks[1])])
        self.layer3 = nn.Sequential(*[L3Block() for x in range(args.blocks[2])])
        self.layer4 = nn.Sequential(*[L4Block() for x in range(args.blocks[3])])


        self.c1to2 = nn.Conv2d(64, 128, (3, 1), stride=(1, 1), padding=(1, 0))
        self.b1to2 = nn.BatchNorm2d(128)
        self.l1tol2 = nn.Sequential(self.c1to2, self.b1to2,nn.ReLU(inplace=True))

        self.c2to3 = nn.Conv2d(128, 200, (1, 1), padding=(3, 0))
        self.b2to3 = nn.BatchNorm2d(200)
        self.l2tol3 = nn.Sequential(self.c2to3, self.b2to3,nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, self.num_cell_types)
        self.flayer = self.final_layer()

    def final_layer(self):
        self.conv3 = nn.Conv2d(200, 200, (7,1), stride =(1,1), padding = (4,0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))


    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)  # batch_size x 4 x 1000 x 1 [4 channels]

        out = self.prelayer(s)
        out = self.layer1(out)
        out = self.layer2(self.l1tol2(out))
        out = self.maxpool1(out)
        out = self.layer3(self.l2tol3(out))
        out = self.maxpool2(out)
        out = self.layer4(out)
        out = self.flayer(out)
        out = self.maxpool3(out)
        out = out.view(-1, 4200)
        conv_out = out
        out = F.dropout(F.relu(self.bn4(self.fc1(out))), p=self.dropout, training=self.training)  # batch_size x 1000
        out = F.dropout(F.relu(self.bn5(self.fc2(out))), p=self.dropout, training=self.training)  # batch_size x 1000
        out = self.fc3(out)
        return out, conv_out


if __name__ == '__main__':
    args = getargs()
    print(args)
 
    model = runner.instantiate_model_stage1(args, Net, model_pipeline)
    di = runner.load_data_iterator_stage1(args)
    runner.run_stage1(model, di, args, model_pipeline)
