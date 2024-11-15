import numpy as np

import random
import torch

from utils.evaluation import evaluator

from utils.utils import AverageMeter
from utils.utils import draw_box_plot_correlation, draw_box_plot_single


def train(args, model, criterion, optimizer, trainloader, panel=None, img_information=True):
    model.train()
    losses = AverageMeter()

    predict_list, traget_list, rmse_list = [], [], []

    for batch_idx, (rna, protein, rna_neighbors, _) in enumerate(trainloader):

        rna, protein, rna_neighbors = rna.cuda(), protein.cuda(), rna_neighbors.cuda()

        ############
        if random.random() > 0.7:
            mask = torch.ones((rna_neighbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).cuda()
            rna_neighbors = rna_neighbors * mask
        ############

        source, target, source_neightbors = rna, protein, rna_neighbors
       
        outputs = model(source, source_neightbors)
      
        loss = criterion(outputs, target) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        predict_list.append(outputs)
        traget_list.append(target)

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, traget_list, training=True, panel=panel)

    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print('Training Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'
                                            .format(pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std))


def test(args, model, testloader, best_pearson, panel=None, last_epoch=False, img_information=True):
    model.eval()

    predict_list, target_list = [], []
    coordinates = []
    
    with torch.no_grad():
        for _, (rna, protein, rna_neighbors, samples) in enumerate(testloader):

            rna, protein, rna_neighbors = rna.cuda(), protein.cuda(), rna_neighbors.cuda()
            
            source, target, source_neightbors = rna, protein, rna_neighbors 
            
            outputs = model(source, source_neightbors)

            predict_list.append(outputs)
            target_list.append(target)
            coordinates += samples

    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list, False, panel, best_pearson)
    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print('Testing Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'
                                            .format(pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std))
    
    # for saving the model
    name = 'pearson:' + str(pearson_mean)[0:5] + '_spearman:' + str(spearman_mean)[0:5]

    if last_epoch == True:
        draw_box_plot_correlation([pearson_sample_list, spearman_sample_list], training=False, results=name, save_dir='plots_last')
        draw_box_plot_single(rmse_list, training=False, results=str(rmse_mean), label=['rmse'], save_dir='plots_last')
        
    return pearson_mean