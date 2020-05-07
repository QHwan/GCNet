from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, dropout):
        super(GCN, self).__init__()

        n_gc_layers = 2
        self.gc_layers = nn.ModuleDict({})
        for i in range(n_gc_layers):
            self.gc_layers['gc{}'.format(i)] = GraphConvolution(n_feat, n_feat)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()
            
            if i == n_gc_layers-1:
                self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)


        self.nn_layers = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, 1)
        )

        self.dropout = dropout

    def forward(self, X, A):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = gc_layer(X, A)
            else:
                X = gc_layer(X)
            graph_layers.append(X)
       
        X = torch.mean(torch.mean(torch.stack(graph_layers), dim=0), dim=1)

        X = self.nn_layers(X)
        return(X)


class GAT(nn.Module):
    def __init__(self, n_node, n_feat, n_hid, n_batch, dropout):
        super(GAT, self).__init__()

        n_gat_layers = 1
        self.gat_layers = nn.ModuleDict({})
        for i in range(n_gat_layers):
            self.gat_layers['gat{}'.format(i)] = GraphAttention(n_node, n_feat, n_feat, n_batch=n_batch)
            self.gat_layers['relu{}'.format(i)] = nn.ReLU()
            
            if i == n_gat_layers-1:
                self.gat_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

        self.dropout = dropout

        self.nn_layers = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, 1)
        )

        self.dropout = dropout

    def forward(self, X, A):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gat_layer) in enumerate(self.gat_layers.items()):
            if key.startswith('gat'):
                X = gat_layer(X, A)
            else:
                X = gat_layer(X)
            graph_layers.append(X)
       
        X = torch.mean(torch.mean(torch.stack(graph_layers), dim=0), dim=1)

        X = self.nn_layers(X)
        return(X)

