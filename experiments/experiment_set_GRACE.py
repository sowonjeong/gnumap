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
from experiments.experiment import experiment


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
parser.add_argument('--n_neighbours', type=int, default=15) #
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
parser.add_argument('--result_file', type=str, default="/results/MVGRL_node_classification.csv")
parser.add_argument('--embeddings', type=str, default="/results/MVGRL_node_classification_embeddings.csv")
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = NormalizeFeatures()
diff_transform = T.Compose([T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out=None,
        diffusion_kwargs=dict(method='ppr', alpha=0.2), sparsification_kwargs=dict(method='threshold', eps=0.01),
        exact=True), T.ToDevice(device)])
diff = None

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
    if args.model in ['MVGRL', 'GNUMAP']:
        diff_dataset = Planetoid(root='Planetoid', name=args.dataset,
                                 transform=diff_transform)
        diff = diff_dataset[0]
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(args.dataset, 'public', transform=transform)
    data = dataset[0]
    if  args.model in ['MVGRL', 'GNUMAP']:
        diff_dataset = Coauthor(args.dataset, 'public', transform=diff_transform)
        diff = diff_dataset[0]
if args.dataset in ['Computers', 'Photo']:
    dataset = Amazon("/Users/ilgeehong/Desktop/SemGCon/", args.dataset, transform=transform)
    data = dataset[0]
    if  args.model in ['MVGRL', 'GNUMAP']:
            diff_dataset = Amazon("/Users/cdonnat/Dp/SemGCon/", args.dataset, transform=diff_transform)
            diff = diff_dataset[0]

dataset_print(dataset)
data_print(data)


file_path = os.getcwd() +  args.result_file

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

results = []

for dim in [16, 32, 64, 128, 256, 512]:
    for proj_hid_dim in [128, 256, 512]:
        for tau in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for edr in [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for fmr in [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                        _, res = experiment(model='GRACE', data=data,
                                   train_data=train_data, val_data=val_data, test_data=test_data,
                                   rand_data = rand_data,
                                   diff = diff, target = target, device=device,
                                   patience=args.patience, epochs=args.epochs,
                                   n_layers=args.n_layers, out_dim=dim,
                                   lr1=args.lr1, lr2=args.lr2, wd1=args.wd1,
                                   wd2=args.wd2, tau=tau, lambd=args.lambd,
                                   min_dist=args.min_dist,
                                   method=args.method, n_neighbours=args.n_neighbours,
                                   beta=args.beta, norm=args.norm, edr=edr, fmr=fmr,
                                   proj=args.proj, pred_hid=args.pred_hid, proj_hid_dim=proj_hid_dim,
                                   dre1=args.dre1, dre2=args.dre2, drf1=args.drf1, drf2=args.drf2)
    results += [res]
    pd.DataFrame(np.array(results),
                 columns =[ 'model', 'method',
                            'dim', 'neighbours', 'n_layers', 'norm','min_dist',
                             'dre1', 'drf1', 'lr', 'edr', 'fmr',
                            'tau', 'lambd','pred_hid,' 'proj_hid_dim',
                            'train_roc', 'train_ap',
                            'test_roc', 'test_ap', 'acc_train', 'val_train', 'acc',
                            'acc_train_default', 'acc_val_default', 'acc_default', 'F1Mi-mean',
                            'F1Mi-std','F1Ma-mean', 'F1Ma-std', 'acc-mean',  'acc-std'] ).to_csv(file_path)
    print(results)
