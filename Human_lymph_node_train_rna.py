from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse

import numpy as np
import os.path as osp
import random


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler


from model.nicheTrans_img import *
from datasets.SMA_data_manager import SMA
from datasets.human_lymph_node_data_manager import Lymph_node

from utils.utils import Logger
from utils.Lymph_node_utils_training import train, test
from utils.utils_dataloader import *


parser = argparse.ArgumentParser(description='Multi-omics translation')

################
# for model
parser.add_argument('--noise_rate', default=0.5, type=float)
parser.add_argument('--dropout_rate', default=0.1, type=float)

parser.add_argument('--n_source', default=3000, type=int)
parser.add_argument('--n_target', default=50, type=int)

# Datasets
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

# Training options
parser.add_argument('--max-epoch', default=40, type=int,
                    help="maximum epochs to run")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")

parser.add_argument('--train-batch', default=32, type=int)
parser.add_argument('--test-batch', default=32, type=int)

# Optimization options
parser.add_argument('--optimizer', default='adam', type=str,
                    help="adam or SGD")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--save-dir', type=str, default='./log', help="manual seed")
parser.add_argument('--eval-step', type=int, default=1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    
    index = len(os.listdir('./log')) + 1
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train_{}.txt'.format(str(index))))
   
    print("==========\nArgs:{}\n==========".format(args))
    
    # create the dataloaders
    dataset = Lymph_node()
    trainloader, testloader = human_node_dataloader(args, dataset)

    # create the model
    source_dimension, target_dimension = dataset.rna_length, dataset.msi_length
    model = NicheTrans(source_length=source_dimension, target_length=target_dimension, noise_rate=args.noise_rate, dropout_rate=args.dropout_rate)
    model = nn.DataParallel(model).cuda()

    # criterion and optimization
    criterion = nn.MSELoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        print('unexpected optimizer')

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # model training and testing
    start_time = time.time()
    best_pearson = 0.0

    for epoch in range(args.max_epoch):
        last_epoch = epoch + 1 == args.max_epoch

        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        ################
        train(args, model, criterion, optimizer, trainloader, dataset.target_panel)
        if args.stepsize > 0: scheduler.step()
        
        if (epoch+1) % args.eval_step == 0:
            print("==> Test")
            pearson = test(args, model, testloader, best_pearson, dataset.target_panel, last_epoch)

        if last_epoch==True:
            torch.save(model.state_dict(), 'last.pth')
        ################
    
    print('Best pearson correlation {:.4f}'.format(best_pearson))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()
