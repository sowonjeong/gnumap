import numpy as np
import torch
import torch.nn as nn
from models.dgi import DGI
from models.mvgrl import MVGRL

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_dgi(hid_dim, out_dim, n_layers, data, patience=20,
              epochs=200, lr=1e-3):
    num_class = int(data.y.max().item()) + 1
    in_dim = data.num_features
    N = data.num_nodes
    cnt_wait = 0
    best = 1e9
    best_t = 0
    ##### Train DGI model #####
    print("=== train DGI model ===")
    model = DGI(in_dim, hid_dim, out_dim, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    loss_fn1 = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
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

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))
    return(model)


def train_mvgrl(hid_dim, out_dim, n_layers, data, patience=20,
              epochs=200, lr=1e-3, wd1=1e-4):
    model = MVGRL(in_dim, out_dim) # hid_dim, , n_layers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
    loss_fn1 = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
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
            torch.save(model.state_dict(), 'best_mvgrl.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_mvgrl.pkl'))
    return(model)
