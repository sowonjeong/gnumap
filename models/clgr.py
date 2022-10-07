import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.baseline_models import GCN, MLP


class CLGR(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, use_mlp=False,
                normalize=True, standardize=True, lambd=1.0, hinge=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau
        self.normalize = normalize
        self.standardize = standardize
        self.lambd=lambd
        self.use_hinge = hinge

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)
        if self.standardize:
            z1 = (h1 - h1.mean(0)) / h1.std(0)
            z2 = (h2 - h2.mean(0)) / h2.std(0)
            return z1, z2
        else:
            return h1, h2

    def sim(self, z1, z2, indices, refl=None):
        if self.normalize:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
        f = lambda x: torch.exp(0.5 *(x-1) / self.tau)
        if indices is not None:
            z2_new = z2[indices,:]
            sim = f(torch.mm(z1, z2_new.t()))
            #diag = f(torch.mm(z1,z2.t()).diag())
        else:
            sim = f(torch.mm(z1, z2.t()))
            #diag = f(torch.mm(z1, z2.t()).diag())
        #sim  = torch.clip(sim , max=100)
        diag = f(sim.diag())
        return sim, diag

    def semi_loss(self, z1, z2, indices):
        N = z1.shape[0]
        criterion = torch.nn.MultiMarginLoss()
        refl_sim, refl_diag = self.sim(z1, z1, indices)
        if not self.use_hinge:
            refl_sim = refl_sim - torch.diagflat(refl_diag)
            refl_sim = torch.clip(refl_sim , min=1e-5)
        between_sim, between_diag = self.sim(z1, z2, indices)
#         if indices is not None:
#             refl_diag_temp = refl_diag.clone()
#             refl_diag_temp[~indices] = 0.0
#             refl_diag_neg = refl_diag_temp.clone()
#         else:
#             refl_diag_temp = refl_diag.clone()
#             refl_diag_neg = refl_diag_temp.clone()
        #print(torch.log(between_diag).mean(), torch.log(between_sim.sum(1) + refl_sim.sum(1)).mean())
        #print("min, max",(between_sim.sum(1) + refl_sim.sum(1)).min(), (between_sim.sum(1) + refl_sim.sum(1) ).max())
        if self.use_hinge:
            x = torch.from_numpy(np.array(range(N)) )#torch.hstack([torch.eye(N), torch.zeros((N,N))])
            preds = torch.hstack([refl_sim, between_sim] )
            semi_loss = criterion(preds, x)
        else:
            semi_loss = -torch.log(between_diag) + torch.log(refl_sim.sum(1) + between_sim.sum(1))
        return semi_loss

    def loss(self, z1, z2, device, k=None, mean=True):
        N = z1.shape[0]
        if k is not None:
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        if not self.use_hinge:
            l1 = self.semi_loss(z1, z2, indices)
            l2 = self.semi_loss(z2, z1, indices)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        else:
            l1 = self.semi_loss(z1, z2, indices)
            l2 = self.semi_loss(z2, z1, indices)
            ret = (l1 + l2) * 0.5

        if self.standardize == False:
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            c1 = c1 / N
            c2 = c2 / N
            iden = torch.tensor(torch.eye(c1.shape[0])).to(device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            lambd = self.lambd
            loss1 = ret + lambd * (loss_dec1 + loss_dec2)
        else:
            loss1 = ret

#         if self.normalize == False:
#             d1 = torch.norm(z1) **2
#             d2 = torch.norm(z2) **2
#             print(d1, d2)
#             lambd = self.lambd
#             loss = loss1 + lambd * (2-d1/N + d2/N)
#         else:
        loss = loss1

        return loss


class SemiGCon(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau = 0.5, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        return z1, z2

    def sim(self, z1, z2, pos_idx):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.tau)
        # if indices is not None:
        #     z2_new = z2[indices,:]
        #     sim = f(torch.mm(z1, z2_new.t()))
        #     diag = f(torch.mm(z1,z2.t()).diag())
        #     sim_pos_temp1 = f(torch.mm(z1, z2.t()))
        #     sim_pos_temp2 = sim_pos_temp1.clone()
        #     sim_pos_temp2[~pos_idx] = 0
        #     sim_pos = sim_pos_temp2.clone()
        #     sim_pos_sum = sim_pos.sum(1)
        # else:
        sim = f(torch.mm(z1, z2.t()))
        diag = f(torch.mm(z1, z2.t()).diag())
        sim_pos_temp1 = sim.clone()
        sim_pos_temp1[~pos_idx] = 0
        sim_pos = sim_pos_temp1.clone()
        sim_pos_sum = sim_pos.sum(1)
        return sim, diag, sim_pos_sum

    def semi_loss(self, data, z1, z2, num_class, train_idx, indices):
        class_idx = []
        for c in range(num_class):
            index = (data.y == c) * train_idx
            class_idx.append(index)
        class_idx = torch.stack(class_idx).bool()
        pos_idx = class_idx[data.y]
        pos_idx[~train_idx] = False
        pos_idx.fill_diagonal_(True)

        refl_sim, refl_diag, refl_pos_sum = self.sim(z1, z1, pos_idx, indices)
        between_sim, _, between_pos_sum = self.sim(z1, z2, pos_idx, indices)
        num_per_class = pos_idx.sum(1)

        # if indices is not None:
        #     refl_diag_temp = refl_diag.clone()
        #     refl_diag_temp[~indices] = 0.0
        #     refl_diag_neg = refl_diag_temp.clone()
        # else:
        refl_diag_temp = refl_diag.clone()
        refl_diag_neg = refl_diag_temp.clone()

        semi_loss = -torch.log(
            (1/(2*num_per_class-1))*(between_pos_sum + refl_pos_sum - refl_diag) / (between_sim.sum(1) + refl_sim.sum(1) - refl_diag_neg)
            )
        return semi_loss

    def loss(self, data, z1, z2, num_class, train_idx, k=None, mean=True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        l1 = self.semi_loss(data, z1, z2, num_class, train_idx, indices)
        l2 = self.semi_loss(data, z2, z1, num_class, train_idx, indices)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
