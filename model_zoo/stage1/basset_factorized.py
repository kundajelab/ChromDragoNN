import sys
import os
sys.path.append('../../')
from utils import data_iterator
from utils.data_utils import *
import utils.runner as runner
from utils.fetch_global_args import stage1_global_argparser
import torch
import torch.nn as nn
import utils.model_pipeline_basset as model_pipeline
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable



"""
Usage: python basset_factorized_ct.py

Notes: This Model implements a factorized basset model as outlined in
https://www.biorxiv.org/content/biorxiv/early/2017/12/05/229385.full.pdf

Factors the first two layers of the 3 convolutional layers of the Basset model:
The factorizations maintain the effective region of influence of the original layers
and do not significantly increase the overall number of network parameters.


Splits training and testing set as specified in @a args, saving checkpoints specified by args.
Specify learning rates, batch


"""

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

        self.layer1 = self.layer_one()
        self.layer2 = self.layer_two()
        self.layer3 = self.layer_three()
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, self.num_cell_types)


    def layer_one(self):
        self.conv1a = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1b = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1c = nn.Conv2d(64, 100, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1d = nn.Conv2d(100, 150, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv1e = nn.Conv2d(150, 300, (7, 1), stride=(1, 1), padding=(3, 0))

        self.bn1a = nn.BatchNorm2d(48)
        self.bn1b = nn.BatchNorm2d(64)
        self.bn1c = nn.BatchNorm2d(100)
        self.bn1d = nn.BatchNorm2d(150)
        self.bn1e = nn.BatchNorm2d(300)

        tmp = nn.Sequential(self.conv1a,self.bn1a, nn.ReLU(inplace= True),
                            self.conv1b,self.bn1b, nn.ReLU(inplace=True),
                            self.conv1c, self.bn1c, nn.ReLU(inplace=True),
                            self.conv1d, self.bn1d, nn.ReLU(inplace=True),
                            self.conv1e, self.bn1e, nn.ReLU(inplace=True))

        return tmp

    def layer_two(self):
        self.conv2a = nn.Conv2d(300, 200, (7,1), stride = (1,1), padding = (3,0))
        self.conv2b = nn.Conv2d(200, 200, (3,1), stride = (1,1), padding = (1, 0))
        self.conv2c = nn.Conv2d(200, 200, (3, 1), stride =(1,1), padding = (1,0))

        self.bn2a = nn.BatchNorm2d(200)
        self.bn2b = nn.BatchNorm2d(200)
        self.bn2c = nn.BatchNorm2d(200)


        tmp = nn.Sequential(self.conv2a,self.bn2a, nn.ReLU(inplace= True),
                            self.conv2b,self.bn2b, nn.ReLU(inplace=True),
                            self.conv2c, self.bn2c, nn.ReLU(inplace=True))


        return tmp



    def layer_three(self):
        self.conv3 = nn.Conv2d(200, 200, (7,1), stride =(1,1), padding = (4,0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)  # batch_size x 4 x 1000 x 1 [4 channels]
        s = self.maxpool1(self.layer1(s)) # batch_size x 300 x 333 x 1
        s = self.maxpool2(self.layer2(s)) # batch_size x 200 x 83 x 1
        s = self.maxpool3(self.layer3(s)) # batch_size x 200 x 21 x 1
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
