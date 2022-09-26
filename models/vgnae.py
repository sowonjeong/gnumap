import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops



class VGCNEncoder(nn.Module): # in_dim, hid_dims, out_dim, normalize=True
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, scaling=1.0,
    activation='relu', slope=.1, device='cpu', normalize=True):
        super(VGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.normalize = normalize
        self.scaling = scaling

        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [torch.nn.Linear(self.input_dim, self.output_dim),
                        torch.nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.propagate = APPNP(K=1, alpha=0)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x, edge_index):
        h = x
        if self.normalize: h = F.normalize(h, p=2, dim=1) * self.scaling
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                #h = self.fc[c](h)
                mu = self.fc[c](h)
                mu = F.dropout(mu, p=0.5, training=self.training)
                if self.normalize: mu = F.normalize(mu, p=2, dim=1) * self.scaling
                mu = self.propagate(mu, edge_index)


                var = self.fc[c + 1](h)
                var = F.dropout(var, p=0.5, training=self.training)
                if self.normalize: var = F.normalize(var, p=2, dim=1) * self.scaling
                var = self.propagate(var, edge_index)

            else:
                h = self.fc[c](h)
                h = F.dropout(h, p=0.5, training=self.training)
                #if self.normalize: h = F.normalize(h, p=2, dim=1) * self.scaling
                h = self.propagate(h, edge_index)
                h = self._act_f[c](h)
        return mu, var



class DeepVGAE(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels,
                 n_layers=2, normalize=True, activation='relu'):
        super(DeepVGAE, self).__init__(encoder=VGCNEncoder(enc_in_channels,
                                                           enc_hidden_channels,
                                                           enc_out_channels,
                                                           n_layers,
                                                           normalize=normalize,
                                                           activation=activation),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, neg_edge_index, **kwargs):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / x.size(0) * self.kl_loss()
        return pos_loss + neg_loss + kl_loss


    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        self.eval()
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
