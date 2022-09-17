import argparse
from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
import matplotlib.pyplot as plt
import pandas as pd
import random, time
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix

sys.path.append('../')
from data_utils import *
from edge_prediction import edge_prediction
from graph_utils import get_weights, transform_edge_weights
from label_classification import label_classification
from models.baseline_models import *
from models.train_models import *
from node_prediction import node_prediction
from train_utils import *
from umap_functions import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MVGRL')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--out_dim', type=int, default=16) #512
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--lr1', type=float, default=1e-3) #
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd1', type=float, default=0.0)
parser.add_argument('--wd2', type=float, default=0.0)
parser.add_argument('--tau', type=float, default=0.5) #
parser.add_argument('--lambd', type=float, default=1e-4) #
parser.add_argument('--min_dist', type=float, default=0.1) #
parser.add_argument('--neighbours', type=int, default=15) #
parser.add_argument('--method', type=str, default='heat') #
parser.add_argument('--norm', type=str, default='normalize') #
parser.add_argument('--beta', type=float, default=1) #
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--edr', type=float, default=0.5)
parser.add_argument('--fmr', type=float, default=0.2)
parser.add_argument('--proj', type=str, default="standard")
parser.add_argument('--training_rate', type=float, default=0.85)
parser.add_argument('--pred_hid', type=int, default=512)
parser.add_argument('--dre1', type=float, default=0.2)
parser.add_argument('--dre2', type=float, default=0.2)
parser.add_argument('--drf1', type=float, default=0.4)
parser.add_argument('--drf2', type=float, default=0.4)
parser.add_argument('--result_file', type=str, default="/results/MVGRL_node_classification")
parser.add_argument('--embeddings', type=str, default="/results/MVGRL_node_classification_embeddings")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = NormalizeFeatures()
diff_transform = T.Compose([T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out=None,
        diffusion_kwargs=dict(method='ppr', alpha=0.2), sparsification_kwargs=dict(method='threshold', eps=0.01),
        exact=True), T.ToDevice(device)])

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
    if args.model in ['MVGRL', 'GNUMAP']:
        diff_dataset = Planetoid(root='Planetoid', name=args.dataset, transform=diff_transform)
        diff = diff_dataset[0]
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(args.dataset, 'public', transform=transform)
    data = dataset[0]
    if args.model in ['MVGRL', 'GNUMAP']:
        diff_dataset = Coauthor(args.dataset, 'public', transform=diff_transform)
        diff = diff_dataset[0]
if args.dataset in ['Computers', 'Photo']:
    dataset = Amazon("/Users/ilgeehong/Desktop/SemGCon/", args.dataset, transform=transform)
    data = dataset[0]
    if args.model in ['MVGRL', 'GNUMAP']:
            diff_dataset = Amazon("/Users/ilgeehong/Desktop/SemGCon/", args.dataset, transform=diff_transform)
            diff = diff_dataset[0]

dataset_print(dataset)
data_print(data)



embeds = None
val_ratio = (1.0 - args.training_rate) / 3
test_ratio = (1.0 - args.training_rate) / 3 * 2
transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                is_undirected=True, split_labels=True)
transform_nodes = RandomNodeSplit(split = 'test_rest',
                                      num_train_per_class = 20,
                                      num_val = 500)
# transform_umap = RandomLinkSplit(num_val=0.4, num_test=0.4,
#                                 is_undirected=True, split_labels=True)
train_data, val_data, test_data = transform(data)
rand_data = transform_nodes(data)
target = -1 * torch.ones(data.num_nodes).long()
target[rand_data.train_mask] = rand_data.y[rand_data.train_mask]

if args.model == 'DGI':
    model = train_dgi(data, hid_dim=args.out_dim, out_dim=args.out_dim,
                      n_layers=args.n_layers,
                      patience=args.patience,
                      epochs=args.epochs, lr=args.lr1)
    embeds = model.get_embedding(data)
elif args.model == 'MVGRL':
    model = train_mvgrl(data, diff=diff, out_dim=args.out_dim,
                        n_layers=args.n_layers,
                        patience=args.patience,
                        epochs=args.epochs, lr=args.lr1, wd=args.wd1,
                        )
    embeds =  model.get_embedding(data, diff)
elif args.model == 'GRACE':
    model =  train_grace(data, channels=args.out_dim, proj_hid_dim=args.out_dim,
                         tau=args.tau,
                         epochs=args.epochs, lr=args.lr1, wd=args.wd1,
                         fmr=args.fmr, edr=args.edr, proj=args.proj)
    embeds = model.get_embedding(data)
elif args.model == 'GNUMAP':
    model =  train_gnumap(data, target=None, dim=args.out_dim, n_layers=args.n_layers,
                          method = args.method,
                          norm=args.norm, neighbours=args.neighbours,
                          beta=args.beta, patience=args.patience, epochs=args.epochs,
                          lr=args.lr1, wd=args.wd1,
                          min_dist=args.min_dist)
    embeds = model(data.x, data.edge_index)
elif args.model == 'semiGNUMAP':
    model =  train_gnumap(data, target=target, dim=args.out_dim, n_layers=args.n_layers,
                          method = args.method,
                          norm=args.norm, neighbours=args.neighbours,
                          beta=args.beta, patience=args.patience, epochs=args.epochs,
                          lr=args.lr1, wd=args.wd1,
                          min_dist=args.min_dist)
    embeds = model(data.x, data.edge_index)
elif args.model == 'CCA-SSG':
    model =  train_cca_ssg(data, channels=args.out_dim,
                           lambd=args.lambd,
                           n_layers=args.n_layers,
                           epochs=args.epochs, lr=args.lr1,
                           fmr=args.fmr, edr=args.edr)
    embeds = model.get_embedding(data)
elif args.model == 'BGRL':
    model =  train_bgrl(data, channels=args.out_dim,
                           lambd=args.lambd,
                           n_layers=args.n_layers,
                           epochs=args.epochs, lr=args.lr1,
                           fmr=args.fmr, edr=args.edr,
                           pred_hid=args.pred_hid,  wd=args.wd1,
                           drf1=args.drf1, drf2=args.drf2, dre1=args.dre1,
                           dre2=args.dre2)
    embeds = model.get_embedding(data)
else:
    raise ValueError("Model unknown!!")



_, res, best_epoch = edge_prediction(embeds.detach(), embeds.shape[1],
                         train_data, test_data, val_data,
                         lr=0.01, wd=1e-4,
                         patience = 20,
                         max_epochs=500)
val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[best_epoch][1], res[best_epoch][2], res[best_epoch][3], res[best_epoch][4], res[best_epoch][5], res[best_epoch][6]
print("done with the edge prediction")

_, nodes_res, best_epoch = node_prediction(embeds.detach(),
                               dataset.num_classes, data.y,
                               rand_data.train_mask, rand_data.test_mask,
                               rand_data.val_mask,
                               lr=args.lr2, wd=args.wd2,
                               patience = 2000,
                               max_epochs=2000)
print("done with the node prediction")

acc_train, val_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3], nodes_res[best_epoch][4]

_, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                               dataset.num_classes, data.y,
                               data.train_mask, data.test_mask,
                               data.val_mask,
                               lr=args.lr2, wd=args.wd2,
                               patience = 2000,
                               max_epochs=2000)
print("done with the second node prediction")
#
# logreg, best_train_acc, best_val_acc, eval_acc, best_epoch2 = node_prediction2(embeds.detach(),
#                                    dataset.num_classes, data.y,
#                                    data.train_mask, data.test_mask, data.val_mask,
#                                    lr=0.01, wd=1e-4,
#                                     max_epochs=2000)
acc_train_default, acc_val_default, acc_default = nodes_res_default[best_epoch][2], nodes_res_default[best_epoch][3], nodes_res_default[best_epoch][4]
other = label_classification(embeds, data.y, 0.05)
print([train_roc, train_ap,
       test_roc, test_ap, acc_train, val_train, acc,
       acc_train_default, acc_val_default, acc_default,
       other['F1Mi'], other['F1Ma'],   other['acc']])

results = [[train_roc, train_ap,
       test_roc, test_ap, acc_train, val_train, acc,
       acc_train_default, acc_val_default, acc_default,
       other['F1Mi'], other['F1Ma'],   other['acc']]]
print(results)
