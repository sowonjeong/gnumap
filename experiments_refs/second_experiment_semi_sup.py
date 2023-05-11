import torch
import copy
from IPython.core.interactiveshell import InteractiveShell
import math, random, torch, collections, time, torch.nn.functional as F
import networkx as nx, matplotlib.pyplot as plt, numpy as np
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from numbers import Number
from sklearn.metrics import accuracy_score
from scipy import optimize
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix

from scipy import optimize


from carbontracker.tracker import CarbonTracker
import sys,os
sys.path.append('../')
from edge_prediction import edge_prediction
from node_prediction import node_prediction
import pandas as pd
from data_utils import *
from io_utils.visualisation import *
from train_utils import *
from umap_functions import *
from models.baseline_models import *
from graph_utils import get_weights, transform_edge_weights
from label_classification import label_classification
dataset_name = 'Cora'
dataset = Planetoid(root='Planetoid', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]
dataset_print(dataset)
data_print(data)




training_rate = 0.85
MAX_EPOCH_EVAL = 100
use_diffusion_weights = False
use_augmented_adjacency = False
use_laplacian = (not use_diffusion_weights) & ( not use_augmented_adjacency)
val_ratio = (1.0 - training_rate) / 3
test_ratio = (1.0 - training_rate) / 3 * 2
lbda = 1e-2

# MIN_DIST = 1e-1  Worked really well,
MIN_DIST = 1e-1
EPS = 1e-3
add_self_loops=True
alpha = 0.5



#### keep very few edges,
#new_data.edge_index =  remove_self_loops(edge_index)[0],
x = np.linspace(0, 2, 300)

def f(x, min_dist):
    y = []
    for i in range(len(x)):
        if(x[i] <= min_dist):
            y.append(1)
        else:
            y.append(np.exp(- x[i] + min_dist))
    return y

dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))
p , _ = optimize.curve_fit(dist_low_dim, x, f(x, MIN_DIST))
a = p[0]
b = p[1]
print("Hyperparameters a =" + str(a) + " and b = " + str(b))

#tracker = CarbonTracker(epochs=MAX_EPOCH_EVAL)
transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                is_undirected=True, split_labels=True)
transform_nodes = RandomNodeSplit(split = 'test_rest',
                                      num_train_per_class = 20,
                                      num_val = 500)
transform_umap = RandomLinkSplit(num_val=0.4, num_test=0.4,
                                is_undirected=True, split_labels=True)
train_data, val_data, test_data = transform(data)
rand_data = transform_nodes(data)



target = -1 * torch.ones(data.num_nodes).long()
target[rand_data.train_mask] = rand_data.y[rand_data.train_mask]
print(target[rand_data.train_mask])
# MIN_DIST = 1e-1  Worked really well


x = np.linspace(0, 2, 300)

def f(x, min_dist):
    y = []
    for i in range(len(x)):
        if(x[i] <= min_dist):
            y.append(1)
        else:
            y.append(np.exp(- x[i] + min_dist))
    return y

dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))
EPS_0 = data.num_edges/ (data.num_nodes ** 2)



p , _ = optimize.curve_fit(dist_low_dim, x, f(x, MIN_DIST))
a = p[0]
b = p[1]
EPS = math.exp(-1.0/b * math.log(1.0/a * (1.0/EPS_0 -1)))
print("Hyperparameters a = " + str(a) + " and b = " + str(b))
lbda = 1e-2
max_epochs = 150

results = []
for dim in [32, 64, 128, 256, 512, 12]:
    for norm in ["normalize", "standardize", "none" ]:
        model = GNN(data.num_features, dim, dim, n_layers=2,
                                normalize=(norm=='normalize'),
                                standardize=(norm=='standardize'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=1e-4)
        for neighbours in [10, 15, 5, 20]:
            for method in ["heat", "power", "laplacian"]:
                edge_index, edge_weights = get_weights(data, neighbours=15, method = 'laplacian', beta=1)
                #### modify with edge weights
                new_data = Data(x=data.x, edge_index=edge_index,
                                y=data.y, edge_attr=edge_weights)
                transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                            is_undirected=True, split_labels=True)
                train_data, val_data, test_data = transform(new_data)
                transform_nodes = RandomNodeSplit(split = 'test_rest',
                                                  num_train_per_class = 20,
                                                  num_val = 500)
                for epoch in range(max_epochs):
                    #tracker.epoch_start()
                    # optimizer = torch.optim.Adam(model.parameters(),
                    #                              lr=1. - epoch/max_epochs,
                    #                              weight_decay=1e-4)
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    row_pos, col_pos =  train_data.pos_edge_label_index
                    index = (row_pos != col_pos)
                    #### need to remove add_self_loops
                    #row_pos, col_pos =  remove_self_loops(train_data.pos_edge_label_index)

                    diff_norm = torch.sum(torch.square(out[row_pos[index]] - out[col_pos[index]]), 1) + MIN_DIST
                    # print(diff_norm.min(), diff_norm.max())
                    # index = torch.where(diff_norm==0)[0]
                    # print(row_pos[index], col_pos[index])

                    q =  torch.pow(1.  + a * torch.exp(b * torch.log(diff_norm)), -1)
                    edge_weights_pos = train_data.edge_attr[:len(train_data.pos_edge_label)][index]
                    edge_weights_pos = fast_intersection(row_pos[index], col_pos[index], edge_weights_pos,
                                                         target, unknown_dist=0.0, far_dist=3.0)
                    loss =  -torch.mean(edge_weights_pos *  torch.log(q))  #- torch.mean((1.-edge_weights_pos) * (  torch.log(1. - q)))
                    #print("loss pos", loss)
                    row_neg, col_neg =  train_data.neg_edge_label_index
                    index_neg = (row_neg != col_neg)
                    diff_norm = torch.sum(torch.square(out[row_neg[index_neg]] - out[col_neg[index_neg]]), 1)+ MIN_DIST
                    q_neg = torch.pow(1.  + a * torch.exp(b * torch.log(diff_norm)), -1)
                    edge_weights_neg = EPS * torch.ones(len(q_neg))
                    edge_weights_neg = fast_intersection(row_neg[index], col_neg[index], edge_weights_neg,
                                                         target, unknown_dist=1.0, far_dist=3.0)
                    loss +=  - torch.mean((1. - edge_weights_neg) * (torch.log(1.- q_neg)  ))
                    loss.backward()
                    optimizer.step()
                    #print("weight", model.fc[0].weight.grad)
                    #tracker.epoch_end()                    print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

                    if epoch % 10 == 0 :
                        print("=== Evaluation ===")
                        embeds = out
                        # plt.figure()
                        # visualize_umap(out, data.y.numpy(), size=30, epoch=None, loss = None)
                        # plt.show()
                        print(embeds)
                        _, res, best_epoch = edge_prediction(embeds.detach(), embeds.shape[1],
                                                 train_data, test_data, val_data,
                                                 lr=0.01, wd=1e-4,
                                                 patience = 20,
                                                 max_epochs=MAX_EPOCH_EVAL)
                        val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[best_epoch][1], res[best_epoch][2], res[best_epoch][3], res[best_epoch][4], res[best_epoch][5], res[best_epoch][6]

                        _, nodes_res, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       rand_data.train_mask, rand_data.test_mask,
                                                       rand_data.val_mask,
                                                       lr=0.01, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)

                        acc_train, val_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3], nodes_res[best_epoch][4]

                        _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       data.train_mask, data.test_mask,
                                                       data.val_mask,
                                                       lr=0.01, wd=1e-4,
                                                       patience = 20,
                                                       max_epochs=MAX_EPOCH_EVAL)
                        acc_train_default, acc_val_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3], nodes_res_default[best_epoch][4]
                        other = label_classification(embeds, data.y, 0.1)
                        print([ method, neighbours, norm, dim, train_roc, train_ap,
                           test_roc, test_ap, acc_train, val_train, acc,
                           acc_train_default, acc_val_default, acc_default, epoch, other['F1Mi'], other['F1Ma'],   other['acc']])
                        results += [[ method, neighbours, norm, dim, train_roc, train_ap,
                           test_roc, test_ap, acc_train, val_train, acc,
                           acc_train_default, acc_val_default, acc_default, epoch, other['F1Mi'], other['F1Ma'],
                           other['acc']]]
                        pd.DataFrame(np.array( results), columns =[ 'method', 'neighbours', 'norm',
                                                                   'dim', 'train_roc', 'train_ap',
                           'test_roc', 'test_ap', 'acc_train', 'val_train', 'acc',
                           'acc_train_default', 'acc_val_default', 'acc_default', 'epoch', 'F1Mi', 'F1Ma','acc']).to_csv("/Users/cdonnat/Downloads/res_second_experiment_semi_sup2.csv")
None;

#tracker.stop()
None;
