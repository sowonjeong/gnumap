import numpy as np
from carbontracker.tracker import CarbonTracker
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix

from models.dgi import DGI
from models.mvgrl import MVGRL
from models.grace import GRACE
from models.baseline_models import GNN
from models.cca_ssg import CCA_SSG
from models.bgrl import BGRL
from models.data_augmentation import *
from scipy import optimize

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from umap_functions import *
from graph_utils import *

def train_dgi(data, hid_dim, out_dim, n_layers, patience=20,
              epochs=200, lr=1e-3, name_file="1"):
    log_dir = '/log_dir/log_dir_DGI_' + str(out_dim)+ '/'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = int(data.y.max().item()) + 1
    in_dim = data.num_features
    N = data.num_nodes
    cnt_wait = 0
    best = 1e9
    best_t = 0
    ##### Train DGI model #####
    print("=== train DGI model ===")
    model = DGI(in_dim, hid_dim, out_dim, n_layers)
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    loss_fn1 = nn.BCEWithLogitsLoss()
    # tracker = CarbonTracker(epochs=epochs, log_dir=log_dir)
    for epoch in range(epochs):
        #tracker.epoch_start()
        model.train()
        optimizer.zero_grad()
        idx = np.random.permutation(N)
        shuf_fts = data.x[idx,:]
        lbl_1 = torch.ones(1, N)
        lbl_2 = torch.zeros(1, N)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(data, shuf_fts)
        loss = loss_fn1(logits, lbl)

        loss.backward()
        optimizer.step()
        #tracker.epoch_end()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd() + '/results/best_dgi_dim' + str(out_dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
    #tracker.stop()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd() + '/results/best_dgi_dim'
                                     + str(out_dim) + '_' + name_file +  '.pkl'))
    return(model)


def train_mvgrl(data, diff, out_dim, n_layers, patience=20,
              epochs=200, lr=1e-3, wd=1e-4, name_file="1"):

    num_class = int(data.y.max().item()) + 1
    in_dim = data.num_features
    N = data.num_nodes
    cnt_wait = 0
    best = 1e9
    best_t = 0

    lbl_1 = torch.ones(N * 2) #sample_size
    lbl_2 = torch.zeros(N * 2) #sample_size
    lbl = torch.cat((lbl_1, lbl_2))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MVGRL(in_dim, out_dim) # hid_dim, , n_layers
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn1 = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # sample_idx = torch.LongTensor(random.sample(node_list, N)) # sample_size
        # sample = data.subgraph(sample_idx)
        # Dsample = diff.subgraph(sample_idx)
        shuf_idx = np.random.permutation(N) #sample_size
        shuf_fts = data.x[shuf_idx,:]
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(data, diff, shuf_fts)  # sample, Dsample
        loss = loss_fn1(logits, lbl)
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd() +
                       '/results/best_mvgrl_dim' + str(out_dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd() + '/results/best_mvgrl_dim' +
                                     str(out_dim) + '_' + name_file +  '.pkl'))
    return(model)


def train_gnumap(data, dim, n_layers=2, target=None,
                 method = 'laplacian',
                 norm='normalize', neighbours=15,
                 beta=1, patience=20, epochs=200, lr=1e-3, wd=1e-4,
                 min_dist=0.1, name_file="1"):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPS_0 = data.num_edges/ (data.num_nodes ** 2)
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
    p , _ = optimize.curve_fit(dist_low_dim, x, f(x, min_dist))
    a = p[0]
    b = p[1]
    EPS = math.exp(-1.0/(2*b) * math.log(1.0/a * (1.0/EPS_0 -1)))
    print("Epsilon is " + str(EPS))
    print("Hyperparameters a = " + str(a) + " and b = " + str(b))

    model = GNN(data.num_features, dim, dim, n_layers=n_layers,
                            normalize=(norm=='normalize'),
                            standardize=(norm=='standardize'))
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=wd)
    best  = 1e9
    edge_index, edge_weights = get_weights(data, neighbours=neighbours,
                                           method = method, beta=beta)
    #### modify with edge weights
    new_data = Data(x=data.x, edge_index=edge_index,
                    y=data.y, edge_attr=edge_weights)
    row_pos, col_pos =  new_data.edge_index
    index = (row_pos != col_pos)
    edge_weights_pos = new_data.edge_attr[index]
    if target is not None:
        edge_weights_pos = fast_intersection(row_pos[index], col_pos[index], edge_weights_pos,
                                             target, unknown_dist=1.0, far_dist=5.0)

    # row_neg, col_neg = negative_sampling(new_data.edge_index)
    # index_neg = (row_neg != col_neg)
    # edge_weights_neg = EPS * torch.ones(len(row_neg))
    # if target is not None:
    #     edge_weights_neg = fast_intersection(row_neg[index_neg], col_neg[index_neg], edge_weights_neg,
    #                                          target, unknown_dist=1.0, far_dist=3.0)
    best_t=0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        #### need to remove add_self_loops
        #row_pos, col_pos =  remove_self_loops(train_data.pos_edge_label_index)

        diff_norm = torch.sum(torch.square(out[row_pos[index]] - out[col_pos[index]]), 1) + min_dist
        # print(diff_norm.min(), diff_norm.max())
        # index = torch.where(diff_norm==0)[0]
        # print(row_pos[index], col_pos[index])

        q =  torch.pow(1.  + a * torch.exp(b * torch.log(diff_norm)), -1)
        #edge_weights_pos = train_data.edge_attr[:len(train_data.pos_edge_label)][index]
        loss =  -torch.mean(edge_weights_pos *  torch.log(q))  #- torch.mean((1.-edge_weights_pos) * (  torch.log(1. - q)))
        #print("loss pos", loss)
        #row_neg, col_neg =  train_data.neg_edge_label_index
        row_neg, col_neg = negative_sampling(new_data.edge_index)
        index_neg = (row_neg != col_neg)
        edge_weights_neg = EPS * torch.ones(len(row_neg))
        diff_norm_neg = torch.sum(torch.square(out[row_neg[index_neg]] - out[col_neg[index_neg]]), 1) + min_dist

        q_neg = torch.pow(1.  + a * torch.exp(b * torch.log(diff_norm_neg)), -1)
        loss +=  - torch.mean((1. - edge_weights_neg) * (torch.log(1.- q_neg)  ))
        loss.backward()
        optimizer.step()
        #print("weight", model.fc[0].weight.grad)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd() + '/results/best_gnumap_'
                                          + str(method) + '_neigh' + str(neighbours)
                                          + '_dim' + str(dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping at epoch {}!'.format(epoch))
            break
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd() + '/results/best_gnumap_' +
                                     str(method) + '_neigh' + str(neighbours)
                                     + '_dim' + str(dim) + '_' + name_file + '.pkl'))
    return(model)


def train_grace(data, channels, proj_hid_dim, n_layers=2, tau=0.5,
                epochs=100, wd=1e-5, lr=1e-3, fmr=0.2, edr =0.5,
                proj="standard"):

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_idx = data.train_mask
        val_idx = data.val_mask
        test_idx = data.test_mask

        in_dim = data.num_features
        hid_dim = channels
        proj_hid_dim = proj_hid_dim
        n_layers = n_layers
        tau = tau

        num_class = int(data.y.max().item()) + 1
        N = data.num_nodes

        ##### Train GRACE model #####
        print("=== train GRACE model ===")
        model = GRACE(in_dim, hid_dim, proj_hid_dim, n_layers, tau)
        model = model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        def train_grace_one_epoch(model, data, fmr, edr, proj):
                model.train()
                optimizer.zero_grad()
                new_data1 = random_aug(data, fmr, edr)
                new_data2 = random_aug(data, fmr, edr)
                new_data1 = new_data1.to(dev)
                new_data2 = new_data2.to(dev)
                z1, z2 = model(new_data1, new_data2)
                loss = model.loss(z1, z2, layer=proj)
                loss.backward()
                optimizer.step()
                return loss.item()
        for epoch in range(epochs):
            loss = train_grace_one_epoch(model, data, fmr,
                                          edr, proj)
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
        return(model)


def train_cca_ssg(data, channels, lambd=1e-5,
                  n_layers=2, epochs=100, lr=1e-3,
                  fmr=0.2, edr =0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_idx = data.train_mask
    val_idx = data.val_mask
    test_idx = data.test_mask

    in_dim = data.num_features
    hid_dim = channels
    out_dim = channels
    n_layers = n_layers

    num_class = int(data.y.max().item()) + 1
    N = data.num_nodes

    class_idx = []
    for c in range(num_class):
        index = (data.y == c) * train_idx
        class_idx.append(index)
    class_idx = torch.stack(class_idx).bool()
    pos_idx = class_idx[data.y]

    ##### Train the SelfGCon model #####
    print("=== train SelfGCon model ===")
    model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, lambd, N, use_mlp=False) #
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    def train_cca_one_epoch(model, data):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, fmr, edr)
        new_data2 = random_aug(data, fmr, edr)
        new_data1 = new_data1.to(device)
        new_data2 = new_data2.to(device)
        z1, z2 = model(new_data1, new_data2)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(epochs):
        loss = train_cca_one_epoch(model, data) #train_semi(model, data, num_per_class, pos_idx)
        # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
    return(model)


def train_bgrl(data, channels, lambd=1e-5,
                  n_layers=2, epochs=100, lr=1e-3,
                  fmr=0.2, edr =0.5, pred_hid=512, wd=1e-5,
                  drf1=0.2, drf2=0.2, dre1=0.4, dre2=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_idx = data.train_mask
    val_idx = data.val_mask
    test_idx = data.test_mask

    in_dim = data.num_features
    hid_dim = channels
    out_dim = channels
    n_layers = n_layers

    num_class = int(data.y.max().item()) + 1
    N = data.num_nodes

    ##### Train the BGRL model #####
    print("=== train BGRL model ===")
    model = BGRL(in_dim, hid_dim, out_dim, n_layers, pred_hid)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay= wd)
    s = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (epochs - 1000))) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = s)

    def train_bgrl_one_epoch(model, data):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, drf1, dre1)
        new_data2 = random_aug(data, drf2, dre2)

        z1, z2, loss = model(new_data1, new_data2)

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_moving_average()

        return loss.item()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = train_bgrl_one_epoch(model, data)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
    return(model)
