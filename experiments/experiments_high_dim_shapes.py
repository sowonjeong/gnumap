import pandas as pd
import argparse
import torch
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
parser.add_argument('--path', type=str, default="")
parser.add_argument('--seed', type=int, default=2022)
args = parser.parse_args()
FILE_NAME =  args.path  + "/high_dim_corrected_results" +str(args.seed) + "_shape_exp_"  + str(args.n_layers) + '_' + str(args.noise) +  ".csv"
DICT_NAME_APP =  "high_dim_corr_" +str(args.seed) + "_new_reduced" + str(args.n_layers) + '_' + str(args.noise)


# parser.add_argument('--embeddings', type=str, default="/results/MVGRL_node_classification_embeddings.csv")

from models.train_models import train_dgi, train_clgr, train_grace, train_cca_ssg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate

np.random.seed(args.seed)

results = []
for exp in range(50):
    for dataset_type in ['moon', 'swissroll', 'circles']:
        for nb_repeats in [1, 10, 30, 50, 100]:
            reg = LogisticRegression(random_state=0)
            parameters = {'C':[0.1, 0.5, 1, 2, 5, 7, 10]}
            svc = SVC()
            clf = GridSearchCV(svc, parameters)
            if dataset_type == 'moon':
                XX = np.zeros((1000, 2 * nb_repeats))
                y = np.zeros((1000, nb_repeats))
                for i in range(nb_repeats):
                #     XXX, yy = make_moons(n_samples=1000, noise=0.05)
                #     XX, y = make_circles(n_samples=1000, noise=0.05, factor=0.4)
                    XXX, yy = make_moons(n_samples=1000, noise=args.noise)
                    XX[:, 2 * i: (2 * (i+1))] = XXX
                    y[:, i]= yy
            elif dataset_type == 'circles':
                XX = np.zeros((1000, 2 * nb_repeats))
                y = np.zeros((1000, nb_repeats))
                for i in range(nb_repeats):
                #     XXX, yy = make_moons(n_samples=1000, noise=0.05)
                #     XX, y = make_circles(n_samples=1000, noise=0.05, factor=0.4)
                    XXX, yy = make_circles(n_samples=1000, noise=args.noise,
                                         factor=0.4)
                    XX[:, 2 * i: (2 * (i+1))] = XXX
                    y[:, i]= yy
            else:
                svr = SVR()
                parameters = {'C':[0.1, 0.5, 1, 2, 5, 7, 10], 'epsilon': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]}
                clf = GridSearchCV(svr, parameters)
                reg = LinearRegression()
                XX = np.zeros((1000, 3 * nb_repeats))
                y = np.zeros((1000, nb_repeats))
                for i in range(nb_repeats):
                #     XXX, yy = make_moons(n_samples=1000, noise=0.05)
                #     XX, y = make_circles(n_samples=1000, noise=0.05, factor=0.4)
                    XXX, yy = make_swiss_roll(n_samples=1000, noise=args.noise)
                    XX[:, 3 * i: (3 * (i+1))] = XXX
                    y[:, i]= yy

            A = kneighbors_graph(XX, 15, mode='distance', include_self=False)
            edge_index, edge_weights = from_scipy_sparse_matrix(A)
            edge_index, edge_weights = to_undirected(edge_index, edge_weights)
            moon_data = Data(x=torch.eye(1000),
                            edge_index=edge_index,
                            edge_weight=edge_weights)

            model = train_dgi(moon_data,3 * nb_repeats, 2* nb_repeats, args.n_layers, patience=20,
                  epochs=1000, lr=0.01, name_file=DICT_NAME_APP)
            out = model.get_embedding(moon_data).numpy()
            u = out
            best_scores = []
            reg_scores = []
            for i in range(nb_repeats):
                u_train, u_test, y_train, y_test = train_test_split(u, y[:, i], test_size=0.25)
                clf.fit(u_train, y_train)
                clf_best = clf.best_estimator_
                clf_best.fit(u_train, y_train)
                best_scores += [clf_best.score(u_test, y_test)]
                cv_results = cross_validate(reg, u, y[:,i], cv=2)
                reg_scores  += [np.mean(cv_results['test_score'])]
            results+= [[exp, dataset_type, args.noise, 'dgi', np.nan, np.nan, np.nan,
                        nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
            pd.DataFrame(np.array(results),
                                    columns=['exp', 'dataset', 'noise', 'method',
                                            'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)


            for edr in [0.5]:
                for tau in [0.5]:
                    if tau > 1e-1:
                        model_grace = train_grace(moon_data,3 * nb_repeats, 1 +  2 * nb_repeats,
                                                  n_layers=args.n_layers, tau=tau,
                                                  fmr=0.2, edr=edr, proj="nonlinear-hid",
                                                  epochs=1000, lr=0.01, name_file=DICT_NAME_APP)
                        out = model_grace.get_embedding(moon_data).numpy()
                        u = out
                        best_scores = []
                        reg_scores = []
                        for i in range(nb_repeats):
                            u_train, u_test, y_train, y_test = train_test_split(u, y[:, i], test_size=0.25)
                            clf.fit(u_train, y_train)
                            clf_best = clf.best_estimator_
                            clf_best.fit(u_train, y_train)
                            best_scores += [clf_best.score(u_test, y_test)]
                            cv_results = cross_validate(reg, u, y[:,i], cv=2)
                            reg_scores  += [np.mean(cv_results['test_score'])]
                        results+= [[exp, dataset_type, args.noise, 'grace', edr, tau,  np.nan,
                                    nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
                                                        'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)

                        model_grace = train_grace(moon_data,3 * nb_repeats,1 + 2 *  nb_repeats,
                                                  n_layers=args.n_layers, tau=tau,
                                                  fmr=0.2, edr=edr, proj="standard",
                                                  epochs=1000, lr=0.01, name_file=DICT_NAME_APP)
                        out = model_grace.get_embedding(moon_data).numpy()
                        u = out
                        best_scores = []
                        reg_scores = []
                        for i in range(nb_repeats):
                            u_train, u_test, y_train, y_test = train_test_split(u, y[:, i], test_size=0.25)
                            clf.fit(u_train, y_train)
                            clf_best = clf.best_estimator_
                            clf_best.fit(u_train, y_train)
                            best_scores += [clf_best.score(u_test, y_test)]
                            cv_results = cross_validate(reg, u, y[:,i], cv=2)
                            reg_scores  += [np.mean(cv_results['test_score'])]
                        results+= [[exp, dataset_type, args.noise, 'grace_std', edr, tau,  np.nan,
                                    nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
                                                        'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)



                        model_grace = train_grace(moon_data,3 * nb_repeats,1 + 2 *  nb_repeats,
                                                  n_layers=args.n_layers, tau=tau,
                                                  fmr=0.2, edr=edr, proj="dbn",
                                                  epochs=1000, lr=0.01, name_file=DICT_NAME_APP)
                        out = model_grace.get_embedding(moon_data).numpy()
                        u = out
                        best_scores = []
                        reg_scores = []
                        for i in range(nb_repeats):
                            u_train, u_test, y_train, y_test = train_test_split(u, y[:, i], test_size=0.25)
                            clf.fit(u_train, y_train)
                            clf_best = clf.best_estimator_
                            clf_best.fit(u_train, y_train)
                            best_scores += [clf_best.score(u_test, y_test)]
                            cv_results = cross_validate(reg, u, y[:,i], cv=2)
                            reg_scores  += [np.mean(cv_results['test_score'])]
                        results+= [[exp, dataset_type, args.noise, 'grace_dbn', edr, tau,  np.nan,
                                    nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
                        pd.DataFrame(np.array(results),
                                                columns=['exp', 'dataset', 'noise', 'method',
                                                        'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
                                                        'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)
#                    model = train_clgr(moon_data,
#                                       channels=2 * nb_repeats, hid_dim=3 * nb_repeats,
#                                       tau=tau, lambd=0.,
#                                       n_layers=args.n_layers, epochs=1000, lr=0.01,
#                                       fmr=0.2, edr =edr, name_file=DICT_NAME_APP,
#                                       device=None,
#                                       normalize=True,
#                                       standardize=True)

#                    out = model.get_embedding(moon_data).numpy()
#                    u = out
#                    best_scores = []
#                    reg_scores = []
#                    for i in range(nb_repeats):
#                        clf.fit(u, y[:,i])
#                        best_scores += [clf.best_score_]
#                        cv_results = cross_validate(reg, u, y[:,i], cv=5)
#                        reg_scores  += [np.mean(cv_results['test_score'])]
#                    results+= [[exp, dataset_type, args.noise, 'clgr', edr, tau, np.nan,
#                                nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
#                    pd.DataFrame(np.array(results),
#                                    columns=['exp', 'dataset', 'noise', 'method',
#                                            'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
#                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)
#
#                    for lambd in [1e-2]:
#                        model = train_clgr(moon_data,
#                                           channels=2 * nb_repeats, hid_dim=3 * nb_repeats, tau=tau, lambd=lambd,
#                                           n_layers=args.n_layers, epochs=1000, lr=0.01,
#                                           fmr=0.2, edr =edr, name_file= DICT_NAME_APP + "regularized",
#                                           device=None,
#                                           normalize=True,
#                                           standardize=False
#                                           )
#
#                        out = model.get_embedding(moon_data).numpy()
#                        u = out
#                        best_scores = []
#                        reg_scores = []
#                        for i in range(nb_repeats):
#                            clf.fit(u, y[:,i])
#                            best_scores += [clf.best_score_]
#                            cv_results = cross_validate(reg, u, y[:,i], cv=5)
#                            reg_scores  += [np.mean(cv_results['test_score'])]
#
#                        results+= [[exp, dataset_type, args.noise, 'clgr_regularized', edr, tau, lambd,
#                                    nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
#                        pd.DataFrame(np.array(results),
#                                    columns=['exp', 'dataset', 'noise', 'method',
#                                            'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
#                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)


                for lambd in [1e-1]:
                        model_cca = train_cca_ssg(moon_data, channels=2 * nb_repeats,
                                                  hid_dim=3 * nb_repeats, lambd=lambd,
                                                  n_layers=args.n_layers,  epochs=1000, lr=0.01,
                                                  fmr=0.2, edr =edr, name_file=DICT_NAME_APP,
                                                  device=None)
                        out = model_cca.get_embedding(moon_data).numpy()
                        u = out
                        best_scores = []
                        reg_scores = []
                        for i in range(nb_repeats):
                            u_train, u_test, y_train, y_test = train_test_split(u, y[:, i], test_size=0.25)
                            clf.fit(u_train, y_train)
                            clf_best = clf.best_estimator_
                            clf_best.fit(u_train, y_train)
                            best_scores += [clf_best.score(u_test, y_test)]
                            cv_results = cross_validate(reg, u, y[:,i], cv=2)
                            reg_scores  += [np.mean(cv_results['test_score'])]

                        results+= [[exp, dataset_type, args.noise, 'cca', edr, tau, lambd,
                                    nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
                        pd.DataFrame(np.array(results),
                                    columns=['exp', 'dataset', 'noise', 'method',
                                            'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
                                            'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)


#                         model_cca = train_gnumap(moon_data, channels=2, hid_dim=3 * nb_repeats,
#                                                   n_layers=2, epochs=1000, lr=1e-2,
#                                                   name_file="test",
#                                                   device=None)
#                         out = model_cca.get_embedding(moon_data).numpy()
#                         u = out
#                         clf.fit(u, y)
#                         cv_results = cross_validate(reg, u, y, cv=5)
#                         results+= [[exp, dataset_type, args.noise, 'gnumap', edr, np.nan, lambd,
#                                     nb_repeats, np.mean(np.array(best_scores)), np.std(np.array(best_scores)),  np.mean(reg_scores), np.std(reg_scores)]]
#                         pd.DataFrame(np.array(results),
#                                     columns=['exp', 'dataset', 'noise', 'method',
#                                             'edr', 'tau', 'lambd', 'nb_repeats','clf_best', 'std_clf_best',
#                                             'linear_score', 'sd_linear_score']).to_csv(FILE_NAME)
