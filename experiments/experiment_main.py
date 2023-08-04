import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix, to_undirected
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sc
import sklearn as sk
import umap
import pickle
import argparse
#from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
import pandas as pd
import random, time
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.datasets import *

sys.path.append('../')
from models.baseline_models import *
from models.train_models import *
from gnumap.umap_functions import *
from graph_utils import *
from experiments.create_dataset import *
from experiments.experiment import *
from metrics.evaluation_metrics import *


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Blob')
parser.add_argument('--filename', type=str, default='test')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--out_dim', type=int, default=2) #512
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
parser.add_argument('--proj', type=str, default="nonlinear-hid")
parser.add_argument('--pred_hid', type=int, default=256)
parser.add_argument('--dre1', type=float, default=0.2)
parser.add_argument('--dre2', type=float, default=0.2)
parser.add_argument('--drf1', type=float, default=0.4)
parser.add_argument('--drf2', type=float, default=0.4)
parser.add_argument('--result_folder', type=str, default="/results/")
parser.add_argument('--seed', type=int, default=12345) 
parser.add_argument('--npoints', type=int, default=500)
parser.add_argument('--num_neighbor', type=int, default=50) # graph construction
parser.add_argument('--radius_knn', type=float, default=0) # graph construction
parser.add_argument('--bw', type=float, default=1.) # graph construction
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


name = args.name

results = []
embeddings = {}

if name in ['Blob', 'Circles', 'Moons', 'Cora', 'Pubmed']:
    classification = True
else: 
    classification = False

X, y_true, G = create_dataset(name, n_samples = 500, n_neighbours = 50,
                              features = 'none', standardize = True, 
                              centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
                              factor = 0.2, noise = 0.05,
                              random_state = args.seed, 
                              radius_knn = args.radius_knn, bw = args.bw, 
                              SBMtype = 'lazy')
new_data = G
for model_name in ['GRACE','DGI','BGRL','CCA-SSG']:
    for gnn_type in ['symmetric', 'RW']:
        for alpha in np.arange(0,1.1,0.1):
            mod, res, out = experiment(model_name, new_data,X,
                        y_true, None,
                        patience=args.patience, 
                        epochs=args.epochs,
                        n_layers=args.n_layers, out_dim=args.out_dim, lr1=args.lr1, lr2=args.lr2, 
                        wd1=args.wd1, wd2=args.wd2, tau=args.tau, lambd=1e-4, min_dist=0.1,
                        method='heat', n_neighbours=15,
                        norm='normalize', edr=args.edr, fmr=args.fmr,
                        proj=args.proj, pred_hid=args.pred_hid, proj_hid_dim=args.pred_hid,
                        dre1=args.dre1, dre2=args.dre2, drf1=args.drf1, drf2=args.drf2,
                        npoints = args.npoints, n_neighbors = args.num_neighbor, classification = classification,
                        densmap = False, random_state = args.seed, n = 15, perplexity = 30, 
                        alpha = alpha, beta = 1.0, gnn_type = gnn_type, 
                        name_file=args.filename,subsampling=None)
            results += [res]
            # out = mod.get_embedding(new_data)
            embeddings[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha)]  =  {
                                            'model': model_name, 
                                            'alpha': alpha,
                                            'gnn_type': gnn_type,   
                                            'embedding' : out,
                                            'alpha': alpha}
file_path = os.getcwd() + '/results/' + name +'_' + str(args.radius_based) + '_gnn_results_' + args.filename + '.csv'

pd.DataFrame(np.array(results),
                columns =['model', 'method', 'dim', 'neighbours', 'n_layers', 'norm','min_dist',
                          'dre1', 'drf1', 'lr', 'edr', 'fmr', 'tau', 'lambd', 'pred_hid', 'proj_hid_dim',
                          'sp', 'acc', 'local', 'density', 'alpha', 'beta', 'gnn_type']).to_csv(file_path)


pickle.dump(embeddings, open(os.getcwd() +'/results/'+name +'_' + str(args.radius_based) + '_gnn_results_' + args.filename +'.pkl', 'wb'))

print(results)
