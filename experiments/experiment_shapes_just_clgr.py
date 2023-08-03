import pandas as pd
import argparse
import torch
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math, random, torch, collections, time, torch.nn.functional as F, networkx as nx, matplotlib.pyplot as plt, numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from IPython.display import clear_output
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

import numpy as np
import scipy as sc
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected


import sys, os
sys.path.append('../../gnumap/')
from models.data_augmentation import *
from scipy import optimize

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from codecarbon import OfflineEmissionsTracker
from umap_functions import *
from graph_utils import *
from sklearn.datasets import make_moons, make_circles, make_swiss_roll

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--noise', type=float, default=0.01)
args = parser.parse_args()
FILE_NAME = "~/Downloads/results_experiments_new_just_CLGR_"  + str(args.n_layers) + '_' + str(args.noise) +  ".csv"
DICT_NAME_APP = "new_just_CLGR_" str(args.n_layers) + '_' + str(args.noise)
# parser.add_argument('--embeddings', type=str, default="/results/MVGRL_node_classification_embeddings.csv")





import torch
import torch.nn as nn
import torch.nn.functional as F
import random



from models.train_models import train_dgi, train_clgr, train_grace, train_cca_ssg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate


lambd = [1e-3, 1e-2, 1e-1, 0.2,0.5, 1.]
tau = [1e-3, 1e-2, 1e-1, 0.2, 0.5,  1.]
results = []
for exp in range(20):
        for dataset_type in ['moon', 'swissroll', 'circles']:
            reg = LogisticRegression(random_state=0)
            parameters = {'C':[0.1, 0.5, 1, 2, 5, 7, 10]}
            svc = SVC()
            clf = GridSearchCV(svc, parameters)
            if dataset_type == 'moon':
                XX, y = make_moons(n_samples=1000, noise=args.noise)
            elif dataset_type == 'circles':
                XX, y = make_circles(n_samples=1000, noise=args.noise,
                                     factor=0.4)
            else:
                svr = SVR()
                parameters = {'C':[0.1, 0.5, 1, 2, 5, 7, 10], 'epsilon': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]}
                clf = GridSearchCV(svr, parameters)
                XX, y = make_swiss_roll(n_samples=1000, noise=args.noise)
                reg = LinearRegression()

            A = kneighbors_graph(XX, 30, mode='distance', include_self=False)
            edge_index, edge_weights = from_scipy_sparse_matrix(A)
            edge_index, edge_weights = to_undirected(edge_index, edge_weights)
            moon_data = Data(x=torch.eye(1000),
                            edge_index=edge_index,
                            edge_weight=edge_weights)


            for edr in [0.5, 0.7, 0.9]:
                for tau in [1e-2, 1e-1, 0.2, 0.5,  1.]:
                    model = train_clgr(moon_data,
                                       channels=2, hid_dim=128, tau=tau, lambd=0.,
                                       n_layers=args.n_layers, epochs=500, lr=0.1,
                                       fmr=0., edr =edr, name_file=DICT_NAME_APP,
                                       device=None,
                                       normalize=True,
                                       standardize=True)

                    out = model.get_embedding(moon_data).numpy()
                    u = out
                    clf.fit(u, y)
                    cv_results = cross_validate(reg, u, y, cv=5)
                    results+= [[exp, dataset_type, args.noise, 'clgr', edr, tau, np.nan,
                                clf.best_score_, np.mean(cv_results['test_score']), np.std(cv_results['test_score'])]]
                    pd.DataFrame(np.array(results),
                                    columns=['exp', 'dataset', 'noise', 'method',
                                            'edr', 'tau', 'lambd', 'clf_best',
                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)

                    for lambd in [1e-3, 1e-2, 1e-1,   1., 10.]:
                        model = train_clgr(moon_data,
                                           channels=2, hid_dim=128, tau=tau, lambd=lambd,
                                           n_layers=args.n_layers, epochs=500, lr=0.01,
                                           fmr=0., edr =edr, name_file= DICT_NAME_APP + "regularized",
                                           device=None,
                                           normalize=True,
                                           standardize=False
                                           )

                        out = model.get_embedding(moon_data).numpy()
                        u = out
                        cv_results = cross_validate(reg, u, y, cv=5)
                        clf.fit(u, y)

                        results+= [[exp, dataset_type, args.noise, 'clgr_regularized', edr, tau, lambd,
                                    clf.best_score_,
                                    np.mean(cv_results['test_score']), np.std(cv_results['test_score'])]]
                        pd.DataFrame(np.array(results),
                                    columns=['exp', 'dataset', 'noise', 'method',
                                            'edr', 'tau', 'lambd', 'clf_best',
                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)



#                         model_cca = train_gnumap(moon_data, channels=2, hid_dim=128,
#                                                   n_layers=2, epochs=1000, lr=1e-2,
#                                                   name_file="test",
#                                                   device=None)
#                         out = model_cca.get_embedding(moon_data).numpy()
#                         u = out
#                         clf.fit(u, y)
#                         cv_results = cross_validate(reg, u, y, cv=5)
#                         results+= [[exp, dataset_type, args.noise, 'gnumap', edr, np.nan, lambd,
#                                     clf.best_score_, np.mean(cv_results['test_score']), np.std(cv_results['test_score'])]]
#                         pd.DataFrame(np.array(results),
#                                     columns=['exp', 'dataset', 'noise', 'method',
#                                             'edr', 'tau', 'lambd', 'clf_best',
#                                             'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)
