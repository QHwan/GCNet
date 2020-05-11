from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution, GraphAttention

def make_nn_layers(n_nlayer, n_hid, n_feat, dropout):
    nn_module = []
    for i in range(n_nlayer):
        if i == 0:
            nn_module.append(nn.Linear(n_feat, n_hid[0]))
        else:
            nn_module.append(nn.Linear(n_hid[i-1], n_hid[i]))
        nn_module.append(nn.ReLU())
    nn_module.append(nn.Dropout(dropout))
    nn_module.append(nn.Linear(n_hid[-1], 1))
    return(nn.Sequential(*nn_module))


class CoreModule(nn.Module):
    def __init__(self, n_node, n_feat, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):
        super(CoreModule, self).__init__()
        self.n_node = n_node
        self.n_feat = n_feat
        self.n_hid = n_hid
        self.n_nlayer = n_nlayer
        self.n_glayer = n_glayer
        self.n_batch = n_batch
        self.dropout = dropout

        self.nn_layers = make_nn_layers(n_nlayer, n_hid, n_feat, dropout)

class GCN(CoreModule):
    def __init__(self, n_node, n_feat, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):
        super().__init__(n_node, n_feat, n_hid,
                         n_nlayer, n_glayer, n_batch, dropout)

        self.gc_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gc_layers['gc{}'.format(i)] = GraphConvolution(n_feat, n_feat, n_node, n_batch=n_batch)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()           
            if i == n_glayer-1:
                self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A, N):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = gc_layer(X, A)
            elif key.startswith('relu'):
                X = gc_layer(X)
                graph_layers.append(X)
            elif key.startswith('dropout'):
                X = gc_layer(X)
       
        #X = torch.mean(torch.mean(torch.stack(graph_layers), dim=0), dim=1)
        X = torch.sum(X, dim=1)
        X = torch.div(X, N.unsqueeze(1))

        X = self.nn_layers(X)
        return(X)


class GAT(CoreModule):
    def __init__(self, n_node, n_feat, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):        
        super().__init__(n_node, n_feat, n_hid,
                         n_nlayer, n_glayer, n_batch, dropout)

        self.gat_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gat_layers['gat{}'.format(i)] = GraphAttention(n_node, n_feat, n_feat, n_batch=n_batch)
            self.gat_layers['relu{}'.format(i)] = nn.ReLU()
            
            if i == n_glayer-1:
                self.gat_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A, N):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gat_layer) in enumerate(self.gat_layers.items()):
            if key.startswith('gat'):
                X = gat_layer(X, A)
            elif key.startswith('relu'):
                X = gat_layer(X)
                graph_layers.append(X)
            elif key.startswith('dropout'):
                X = gat_layer(X)
       
        X = torch.mean(torch.stack(graph_layers), dim=0)
        X = torch.sum(X, dim=1)
        X = torch.div(X, N.unsqueeze(1))

        X = self.nn_layers(X)
        return(X)



class GCN_Gate(CoreModule):
    def __init__(self, n_node, n_feat, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):
        super().__init__(n_node, n_feat, n_hid,
                         n_nlayer, n_glayer, n_batch, dropout)

        self.z = Parameter(torch.ones(n_glayer))
        self.z.data.uniform_(-1, 1)

        self.gc_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gc_layers['gc{}'.format(i)] = GraphConvolution(n_feat, n_feat, n_node, n_batch=n_batch)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()           
            if i == n_glayer-1:
                self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A):
        graph_layers = []
        graph_layers.append(X)

        idx = 0
        for key, gc_layer in self.gc_layers.items():
            if key.startswith('gc'):
                X = gc_layer(X, A)
            elif key.startswith('relu'):
                X = gc_layer(X)
                X_gate = self.z[idx] * X + (1 - self.z[idx]) * graph_layers[idx]
                graph_layers.append(X_gate)
                idx += 1
            elif key.startswith('dropout'):
                X = gc_layer(X)
       
        X = torch.mean(torch.mean(torch.stack(graph_layers), dim=0), dim=1)

        X = self.nn_layers(X)
        return(X)