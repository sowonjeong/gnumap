import pandas as pd
import argparse
import torch
import time
import math, random, torch, collections, time, torch.nn.functional as F, networkx as nx, numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv
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
parser.add_argument('--wrong_dim', type=int, default=0)
parser.add_argument('--path', type=str, default="")
args = parser.parse_args()

if args.wrong_dim > 0:
    dim = int(args.wrong_dim)
else:
    dim = 2

FILE_NAME =  args.path  + "/results_15nn_corrected_bigger_additional"  + str(dim) +"_"  + str(args.n_layers) + '_' + str(args.noise) +  ".csv"
DICT_NAME_APP =  "shapes_15nn_corrected_bigger_additional" + str(dim) +"_" + str(args.n_layers) + '_' + str(args.noise)
# parser.add_argument('--embeddings', type=str, default="/results/MVGRL_node_classification_embeddings.csv")

from models.train_models import train_dgi, train_clgr, train_grace, train_cca_ssg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
results = []
np.random.seed(seed=1234)
for exp in range(50):
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

            A = kneighbors_graph(XX, 15, mode='distance', include_self=False)
            edge_index, edge_weights = from_scipy_sparse_matrix(A)
            edge_index, edge_weights = to_undirected(edge_index, edge_weights)
            moon_data = Data(x=torch.eye(1000),
                            edge_index=edge_index,
                            edge_weight=edge_weights)


            for edr in [0.5]:
                for tau in [0.7, 1.]:
                    if tau > 0:
                        start_time = time.time()
                        model_grace = train_grace(moon_data, 128, dim, args.n_layers,
                                                  tau=tau,
                                                   proj="nonlinear-hid",
                                                   epochs=500, lr=0.01, fmr=0.2, edr =0.5, name_file=DICT_NAME_APP)
                        out = model_grace.get_embedding(moon_data).numpy()
                        tot_time = time.time() - start_time
                        u = out
                        u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf.fit(u_train, y_train)
                        #u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf_best = clf.best_estimator_
                        clf_best.fit(u_train, y_train)
                        cv_results = cross_validate(reg, u, y, cv=4)
                        results+= [[exp, dataset_type, args.noise, 'grace', edr, tau,  np.nan,
                                     clf_best.score(u_test, y_test), np.mean(cv_results['test_score']), np.std(cv_results['test_score']), tot_time]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'clf_best',
                                                        'linear_score', 'sd_linear_score', 'time']).to_csv(FILE_NAME)


                        start_time = time.time()
                        model_grace = train_grace(moon_data, 128, dim, args.n_layers,
                                                  tau=tau,
                                                   proj="dbn",
                                                   epochs=500, lr=0.01, fmr=0.2, edr =0.5, name_file=DICT_NAME_APP)
                        out = model_grace.get_embedding(moon_data).numpy()
                        tot_time = time.time() - start_time
                        u = out
                        u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf.fit(u_train, y_train)
                        #u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf_best = clf.best_estimator_
                        clf_best.fit(u_train, y_train)
                        cv_results = cross_validate(reg, u, y, cv=4)
                        results+= [[exp, dataset_type, args.noise, 'grace_dbn', edr, tau,  np.nan,
                                     clf_best.score(u_test, y_test), np.mean(cv_results['test_score']), np.std(cv_results['test_score']), tot_time]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'clf_best',
                                                        'linear_score', 'sd_linear_score', 'time']).to_csv(FILE_NAME)

                        start_time = time.time()
                        model_grace = train_grace(moon_data, 128, dim, args.n_layers,
                                                  tau=tau,
                                                   proj="standard",
                                                   epochs=500, lr=0.01, fmr=0.2,
                                                   edr =0.5, name_file=DICT_NAME_APP + 'standard')
                        out = model_grace.get_embedding(moon_data).numpy()
                        tot_time = time.time() - start_time
                        u = out
                        u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf.fit(u_train, y_train)
                        #u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25)
                        clf_best = clf.best_estimator_
                        clf_best.fit(u_train, y_train)
                        cv_results = cross_validate(reg, u, y, cv=4)
                        results+= [[exp, dataset_type, args.noise, 'grace_std', edr, tau,  np.nan,
                                     clf_best.score(u_test, y_test), np.mean(cv_results['test_score']), np.std(cv_results['test_score']), tot_time]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'clf_best',
                                                        'linear_score', 'sd_linear_score', 'time']).to_csv(FILE_NAME)


