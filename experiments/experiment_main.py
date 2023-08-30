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
# from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx, numpy as np
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
parser.add_argument('--name', type=str, default='Circles')
parser.add_argument('--filename', type=str, default='test')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--hid_dim', type=int, default=2)  # 512
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--a', type=float, default=1.)  # data construction
parser.add_argument('--b', type=float, default=1.)  # data construction
parser.add_argument('--radius_knn', type=float, default=0.1)  # graph construction
parser.add_argument('--bw', type=float, default=1.)  # graph construction
parser.add_argument('--seed', type=float, default=1)
parser.add_argument('--jcsv', type=float, default=True)  # make csv?
parser.add_argument('--jm', nargs='+', default=['DGI', 'BGRL', 'GRACE','GNUMAP','CCA-SSG', 'SPAGCN',
                                                'UMAP', 'DenseMAP',
                                                 'PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE'],
                     help='List of models to run')
args = parser.parse_args()

import logging
# Configure logging settings
logging.basicConfig(filename='viz.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_next_file_path(base_path):
    """
    Returns the next available file path by appending an index to the base path.
    For example, if "results_0.csv" exists, it will return "results_1.csv", and so on.
    """
    counter = 0
    # Split the base_path into name and extension
    name, ext = os.path.splitext(base_path)

    # The first file path to try is the base_path itself
    next_path = base_path

    while os.path.exists(next_path):
        next_path = f"{name}_{counter}{ext}"
        counter += 1

    return next_path


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

name = args.name
results = {}

X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name, n_samples=1000,
                                                          features='none', standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise,
                                                          radius_knn=args.radius_knn, bw=args.bw,
                                                          SBMtype='lazy',
                                                          a=args.a,
                                                          b=args.b)


def visualize_dataset(X_ambient, cluster_labels, title, save_path=True):
    plt.figure()
    plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()


visualize_dataset(X_manifold, cluster_labels, title=args.name,
                      save_path=os.getcwd() + '/results/' + "gt_manifold_" + args.name + ".png")
visualize_dataset(X_ambient, cluster_labels, title=args.name,
                      save_path=os.getcwd() + '/results/' + "gt_ambient_" + args.name + ".png")

logging.info('------------------ START EXPERIMENT ---------------------')
for model_name in args.jm:
    if model_name in ['DGI','BGRL']:
        for alpha in np.arange(0, 1.1, 0.2):
            for beta in np.arange(0, 1.1, 0.2):
                for gnn_type in ['symmetric']:
                    mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                               patience=20, epochs=args.epoch,
                                               n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                               hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                               tau=np.nan, lambd=np.nan, alpha=alpha, beta=beta,
                                               gnn_type=gnn_type, dataset=args.name)
        results[f"{name}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}"] = res


    elif model_name in ['GRACE','CCA-SSG', 'SPAGCN','GNUMAP']:
        for gnn_type in ['symmetric', 'RW']:
            for alpha in np.arange(0, 1.1, 0.1):
                if model_name == "GRACE":
                    lambd = 1e-4
                    beta = 1
                    for tau in [0.1, 0.2, 0.5, 1., 10]:
                        mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                   patience=20, epochs=args.epoch,
                                                   n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                   hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                                   tau=tau, lambd=lambd, min_dist=1e-3, edr=0.2, fmr=0.2,
                                                   proj="standard", pred_hid=args.hid_dim, n_neighbors=15,
                                                   random_state=42, perplexity=30, alpha=alpha, beta=beta,
                                                   gnn_type=gnn_type,
                                                   name_file="logsGRACE " + name, dataset=args.name)
                        results[f"{name}_{model_name}_{gnn_type}_{tau}_alpha_{alpha}_beta_{beta}lambd_{lambd}"] = res
                elif model_name == 'CCA-SSG':
                    beta = 1
                    for tau in [0.1, 0.2, 0.5, 1., 10]:
                        for lambd in [1e-3, 5 *1e-2, 1e-2, 5 *1e-1, 1e-1, 1.]:
                            mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                       patience=20, epochs=args.epoch,
                                                       n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                       hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                                       tau=tau, lambd=lambd, min_dist=1e-3, edr=0.2, fmr=0.2,
                                                       proj="standard", pred_hid=args.hid_dim,
                                                       n_neighbors=15,
                                                       random_state=42, perplexity=30, alpha=alpha, beta=beta,
                                                       gnn_type=gnn_type,
                                                       name_file="logsCCA-SSG " + name, dataset=args.name)
                            results[f"{name}_{model_name}_{gnn_type}_{tau}_alpha_{alpha}_beta_{beta}lambd_{lambd}"] = res
                elif model_name == 'SPAGCN':
                    beta = 1
                    for lambd in [1e-3, 5 * 1e-2, 1e-2, 5 * 1e-1, 1e-1, 1.]:
                        mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                     patience=20, epochs=args.epoch,
                                                     n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                     hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                                     tau=np.nan, lambd=lambd, alpha=alpha, beta=beta,
                                                     gnn_type=gnn_type, dataset=args.name)
                        results[f"{name}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}lambd_{lambd}"] = res
                elif model_name == 'GNUMAP':
                    beta = 1
                    for tau in [0.01, 0.1, 0.2, 0.5, 1., 10]:
                        for lambd in [1e-3, 5 * 1e-2, 1e-2, 5 * 1e-1, 1e-1, 1.]:
                            mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                       patience=20, epochs=args.epoch,
                                                       n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                       hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                                       tau=tau, lambd=lambd, min_dist=1e-3, edr=0.2, fmr=0.2,
                                                       proj="standard", pred_hid=args.hid_dim, n_neighbors=np.nan,
                                                       random_state=42, perplexity=30, alpha=alpha, beta=beta,
                                                       gnn_type=gnn_type,
                                                       name_file="logsGNUMAP " + name, dataset=args.name)
                            results[f"{name}_{model_name}_{gnn_type}_{tau}_alpha_{alpha}_beta_{beta}lambd_{lambd}"] = res
                else:
                    raise ValueError('Invalid model name')

    # 'UMAP', 'DenseMAP'
    elif model_name in ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE','UMAP', 'DenseMAP']:
        mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                   out_dim=X_manifold.shape[1], dataset=args.name)
        results[name + '_' + model_name] = res

if args.jcsv:
    file_path = os.getcwd() + '/results/' + name + '_' + str(args.radius_knn) + '_gnn_results_0.csv'
    pd.DataFrame.from_dict(results, orient='index').to_csv(get_next_file_path(file_path))

logging.info('------------------- END EXPERIMENT ------s----------------')