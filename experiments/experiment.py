import argparse
from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
#import matplotlib.pyplot as plt
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




def experiment(model, data,
               train_data, val_data, test_data,
               rand_data,
               diff, target, device,
               patience=20, epochs=500,
               n_layers=2, out_dim=16, lr1=1e-3, lr2=1e-2, wd1=0.0,
               wd2=0.0, tau=0.5, lambd=1e-4, min_dist=0.1,
               method='heat', n_neighbours=15,
               beta=1, norm='normalize', edr=0.5, fmr=0.2,
               proj="standard", pred_hid=512, proj_hid_dim=512,
               dre1=0.2, dre2=0.2, drf1=0.4, drf2=0.4,
               name_file="1"):

    num_classes = int(data.y.max().item()) + 1
    if model == 'DGI':
        model = train_dgi(data, hid_dim=out_dim, out_dim=out_dim,
                          n_layers=n_layers,
                          patience=patience,
                          epochs=epochs, lr=lr1,
                          name_file=name_file)
        embeds = model.get_embedding(data)
    elif model == 'MVGRL':
        model = train_mvgrl(data, diff=diff, out_dim=out_dim,
                            n_layers=n_layers,
                            patience=patience,
                            epochs=epochs, lr=lr1, wd=wd1,
                            name_file=name_file)
        embeds =  model.get_embedding(data, diff)
    elif model == 'GRACE':
        model =  train_grace(data, channels=out_dim, proj_hid_dim=out_dim,
                             tau=tau,
                             epochs=epochs, lr=lr1, wd=wd1,
                             fmr=fmr, edr=edr, proj=proj)
        embeds = model.get_embedding(data)
    elif model == 'GNUMAP':
        model =  train_gnumap(data, target=None, dim=out_dim, n_layers=n_layers,
                              method = method,
                              norm=norm, neighbours=n_neighbours,
                              beta=beta, patience=patience, epochs=epochs,
                              lr=lr1, wd=wd1,
                              min_dist=min_dist,
                              name_file=name_file)
        embeds = model(data.x, data.edge_index)
    elif model == 'semiGNUMAP':
        model =  train_gnumap(data, target=target, dim=out_dim, n_layers=n_layers,
                              method = method,
                              norm=norm, neighbours=n_neighbours,
                              beta=beta, patience=patience, epochs=epochs,
                              lr=lr1, wd=wd1,
                              min_dist=min_dist,
                              name_file=name_file)
        embeds = model(data.x, data.edge_index)
    elif model == 'CCA-SSG':
        model =  train_cca_ssg(data, channels=out_dim,
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr1,
                               fmr=fmr, edr=edr)
        embeds = model.get_embedding(data)
    elif model == 'BGRL':
        model =  train_bgrl(data, channels=out_dim,
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr1,
                               fmr=fmr, edr=edr,
                               pred_hid=pred_hid,  wd=wd1,
                               drf1=drf1, drf2=drf2, dre1=dre1,
                               dre2=dre2)
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
                                   num_classes, data.y,
                                   rand_data.train_mask, rand_data.test_mask,
                                   rand_data.val_mask,
                                   lr=lr2, wd=wd2,
                                   patience = 2000,
                                   max_epochs=2000)
    print("done with the node prediction")

    acc_train, val_train, acc = nodes_res[best_epoch][2], nodes_res[best_epoch][3], nodes_res[best_epoch][4]

    _, nodes_res_default, best_epoch = node_prediction(embeds.detach(),
                                   num_classes, data.y,
                                   data.train_mask, data.test_mask,
                                   data.val_mask,
                                   lr=lr2, wd=wd2,
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

    results = [
                 method,
                 out_dim,
                 n_neighbours,
                 n_layers,
                 norm,
                 min_dist,
                 dre1,
                 drf1,
                 lr1,
                 edr,
                 fmr,
                 tau,
                 lambd,
                 pred_hid,
                 proj_hid_dim,
                 train_roc, train_ap,
                 test_roc, test_ap, acc_train, val_train, acc,
                 acc_train_default, acc_val_default, acc_default,
                 other['F1Mi']['mean'],
                 other['F1Mi']['std'],
                 other['F1Ma']['mean'],
                 other['F1Ma']['std'],
                 other['acc']['mean'],
                 other['acc']['std']]
    return(model, results)
