import sys, os
import logging
logging.basicConfig(filename='expmain.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
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
parser.add_argument('--name_dataset', type=str, default='Swissroll')
parser.add_argument('--filename', type=str, default='test')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--noise', type=float, default=0)

parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--edr', type=float, default=0.1)
parser.add_argument('--fmr', type=float, default=0.9)
parser.add_argument('--n_neighbours', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--hid_dim', type=int, default=256)

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--a', type=float, default=1.)  # data construction
parser.add_argument('--b', type=float, default=1.)  # data construction
parser.add_argument('--radius_knn', type=float, default=0)  # graph construction
parser.add_argument('--bw', type=float, default=1.)  # graph construction
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_img', type=bool, default=True)
parser.add_argument('--jcsv', type=float, default=False)  # make csv?
parser.add_argument('--jm', nargs='+', default=['SPAGCN'
                                                ],
                    help='List of models to run')
args = parser.parse_args()

save_img = args.save_img

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

name_file = args.name_dataset + "_" + args.filename
results = {}

logging.info('STARTING EXPERIMENT')
X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name_dataset, n_samples=1000,
                                                          n_neighbours=args.n_neighbours,standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise,
                                                          radius_knn=0, bw=args.bw,
                                                          SBMtype='lazy',
                                                          a=args.a,
                                                          b=args.b)


def visualize_dataset(X_ambient, cluster_labels, title, save_img, save_path):
    if save_img:
        plt.figure()
        plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)
        plt.title(title)
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        pass


# Visualize loss + does logging for each file
def viz_loss(loss_values, file_name, epoch = args.epoch):
    logging.info(str(file_name))
    while len(loss_values) < epoch:
        loss_values.append(np.nan)
    loss_df.loc[str(file_name)] = loss_values


visualize_dataset(X_manifold, cluster_labels, title=args.name_dataset, save_img=save_img,
                  save_path=os.getcwd() + '/results/' + "gt_manifold_" + name_file + ".png")
visualize_dataset(X_ambient, cluster_labels, title=args.name_dataset, save_img=save_img,
                  save_path=os.getcwd() + '/results/' + "gt_ambient_" + name_file + ".png")

for model_name in args.jm:
    if model_name == 'SPAGCN':
        loss_df = pd.DataFrame(columns=np.arange(1, 10001))
    else:
        loss_df = pd.DataFrame(columns=np.arange(1,args.epoch+1))
    if model_name in ['DGI', 'GNUMAP']:
        for alpha in np.arange(0,1.1,0.5):
            for beta in np.arange(0,1.1,0.5):
                for gnn_type in ['symmetric','RW']:
                    file_name = f"{args.name_dataset}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}_"+name_file
                    if model_name == 'DGI':
                        mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                                patience=20, epochs=args.epoch,
                                                                n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                                hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                                                tau=np.nan, lambd=np.nan, alpha=alpha, beta=beta,
                                                                gnn_type=gnn_type, dataset=args.name_dataset,
                                                                name_file=name_file,
                                                                save_img=save_img)
                    elif model_name == 'GNUMAP':
                        mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                                patience=20, epochs=args.epoch,
                                                                n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                                hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                                                min_dist=1e-3, edr=0.1, fmr=0.1,
                                                                proj="standard", pred_hid=args.hid_dim,
                                                                n_neighbors=args.n_neighbours,
                                                                random_state=42, perplexity=30, alpha=alpha, beta=beta,
                                                                gnn_type=gnn_type,
                                                                name_file="logsGNUMAP " + name_file,
                                                                dataset=args.name_dataset,
                                                                save_img=save_img)
                    viz_loss(loss_values=loss_values, file_name=file_name)
                    results[file_name] = res if res is not None else {}
        loss_df.to_csv('results/loss/'+args.name_dataset+'_'+model_name+'.csv',index=True)

    elif model_name in ['BGRL', 'CCA-SSG']:
        for gnn_type in ['symmetric']:
            for alpha in [0.5]:
                for beta in [1]:
                    for lambd in [1e-3, 1e-2, 1e-1, 1.]:
                        if model_name == 'BGRL':
                            file_name = f"{args.name_dataset}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}_lambd_{lambd}_" + name_file
                            mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold,
                                                                    cluster_labels,
                                                                    patience=20, epochs=args.epoch,
                                                                    n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                                    hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                                                    tau=np.nan, lambd=lambd, alpha=alpha, beta=beta,
                                                                    gnn_type=gnn_type, dataset=args.name_dataset,
                                                                    name_file=name_file,
                                                                    save_img=save_img)
                            viz_loss(loss_values=loss_values, file_name=file_name)
                            results[file_name] = res if res is not None else {}
                        elif model_name == 'CCA-SSG':
                            for (fmr, edr) in [(args.fmr, args.edr)]:
                                file_name = f"{args.name_dataset}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}" \
                                            f"_lambd_{lambd}_fmr_{fmr}_edr_{edr}" + name_file
                                mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold,
                                                                        cluster_labels,
                                                                        patience=20, epochs=args.epoch,
                                                                        n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                                        hid_dim=args.hid_dim, lr=1e-4, wd=0.0,
                                                                        lambd=lambd, min_dist=1e-3, edr=edr, fmr=fmr,
                                                                        proj="standard", pred_hid=args.hid_dim,
                                                                        n_neighbors=args.n_neighbours,
                                                                        random_state=42, perplexity=30, alpha=alpha,
                                                                        beta=beta,
                                                                        gnn_type=gnn_type,
                                                                        name_file="logsCCA-SSG " + name_file,
                                                                        dataset=args.name_dataset,
                                                                        save_img=save_img)
                                viz_loss(loss_values=loss_values, file_name=file_name)
                                results[file_name] = res if res is not None else {}
        loss_df.to_csv('results/loss/'+f"neigh{args.n_neighbours}_lambd_{lambd}_fmr_{fmr}_edr_{edr}.csv",index=True)

    elif model_name == 'GRACE':
        for gnn_type in ['symmetric', 'RW']:
            for alpha in np.arange(0,1.1, 0.5):
                for beta in np.arange(0,1.1, 0.5):
                    for tau in [0.1, 0.2, 0.5, 1., 10]:
                        for (fmr, edr) in [(0, 0.8), (0.2, 0.2)]:
                            file_name = f"{args.name_dataset}_{model_name}_{gnn_type}_alpha_{alpha}_beta_{beta}" \
                                        f"_tau_{tau}_fmr_{fmr}_edr_{edr}"+name_file
                            mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                                    patience=20, epochs=args.epoch,
                                                                    n_layers=args.n_layers, out_dim=3,
                                                                    hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                                                    tau=tau, min_dist=1e-3, edr=0.2, fmr=0.2,
                                                                    proj="standard", pred_hid=args.hid_dim, n_neighbors=args.n_neighbours,
                                                                    random_state=42, perplexity=30, alpha=alpha, beta=beta,
                                                                    gnn_type=gnn_type,
                                                                    name_file="logsGRACE " + name_file,
                                                                    dataset=args.name_dataset, save_img=save_img)
                            viz_loss(loss_values=loss_values, file_name=file_name)
                            results[file_name] = res if res is not None else {}
        loss_df.to_csv('results/loss/'+args.name_dataset+'_'+model_name+'.csv',index=True)

    elif model_name == 'SPAGCN':
        for alpha in [0.5]:
            file_name = f"{args.name_dataset}_{model_name}_alpha_{alpha}_"+name_file
            mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                    patience=20, epochs=args.epoch,
                                                    n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                                    hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                                    alpha=alpha, dataset=args.name_dataset,
                                                    name_file="logs-Spagcn" + name_file, save_img=save_img)
            viz_loss(loss_values=loss_values, file_name=file_name)
            results[file_name] = res if res is not None else {}
        loss_df.to_csv('results/loss/'+args.name_dataset+'_'+model_name+'.csv',index=True)

    elif model_name in ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP']:
        mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                                out_dim=X_manifold.shape[1], dataset=args.name_dataset,
                                                save_img=save_img, name_file=name_file)
        results[args.name_dataset + '_' + model_name+ '_' + name_file] = res if res is not None else {}

    else:
        raise ValueError('Invalid model name')

if args.jcsv:
    file_path = os.getcwd() + '/results/' + args.name_dataset + '_' + str(args.radius_knn) + \
                '_gnn_results_' + str(args.seed) + '_' + name_file + '.csv'
    pd.DataFrame.from_dict(results, orient='index').to_csv(file_path)

logging.info('ENDING EXPERIMENT')
