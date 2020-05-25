from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *

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
    def __init__(self, params):
        super(CoreModule, self).__init__()
        self.n_node = params['n_node']
        self.n_node_fea = params['n_node_fea']
        self.n_edge_fea = params['n_edge_fea']
        self.n_hid = params['n_hid']
        self.n_nlayer = params['n_nlayer']
        self.n_glayer = params['n_glayer']
        self.n_batch = params['n_batch']
        self.dropout = params['dropout']

        self.nn_layers = make_nn_layers(self.n_nlayer, self.n_hid, self.n_node_fea, self.dropout)

class GCN(CoreModule):
    def __init__(self, params):
        super().__init__(params)

        self.gc_layers = nn.ModuleDict({})
        for i in range(self.n_glayer):
            self.gc_layers['gc{}'.format(i)] = GraphConvolution(self.n_node_fea, self.n_node_fea, self.n_node,
                                                                n_batch=self.n_batch)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()           
            #if i == n_glayer-1:
            #    self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A, E, N):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = gc_layer(X, A)
            elif key.startswith('relu'):
                X = X + gc_layer(X)  ## residual network
                graph_layers.append(X)
            elif key.startswith('dropout'):
                X = gc_layer(X)
        X = torch.sum(X, dim=1)
        X = torch.div(X, N.unsqueeze(1))

        X = self.nn_layers(X)
        return(X)

class MPNN(CoreModule):
    def __init__(self, n_node, n_node_fea, n_edge_fea, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):
        super().__init__(n_node, n_node_fea, n_edge_fea, n_hid,
                         n_nlayer, n_glayer, n_batch, dropout)

        self.gc_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gc_layers['gc{}'.format(i)] = MessagePassing(n_node_fea, n_edge_fea, n_node, n_batch=n_batch)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()           
            #if i == n_glayer-1:
            #    self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A, E, N):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = gc_layer(X, A, E, N)
            elif key.startswith('relu'):
                X = X + gc_layer(X)  ## residual network
                #X = gc_layer(X)
                graph_layers.append(X)
            elif key.startswith('dropout'):
                X = gc_layer(X)
        #print(X.shape, N.shape)
        #X = torch.stack(graph_layers, dim=0)
        #X = torch.mean(X, dim=0)
        X = torch.sum(X, dim=1)
        X = torch.div(X, N.unsqueeze(1))

        X = self.nn_layers(X)
        return(X)


class GAT_MPNN(CoreModule):
    def __init__(self, n_node, n_node_fea, n_edge_fea, n_hid,
                 n_nlayer, n_glayer, n_batch, dropout):
        super().__init__(n_node, n_node_fea, n_edge_fea, n_hid,
                         n_nlayer, n_glayer, n_batch, dropout)

        self.gc_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gc_layers['gc{}'.format(i)] = AttentionMessagePassing(n_node_fea, n_edge_fea, n_node, n_batch=n_batch)
            if i == n_glayer-1:
                self.gc_layers['dropout{}'.format(i)] = nn.Dropout(dropout)

    def forward(self, X, A, E, N):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = X + gc_layer(X, A, E, N)
                graph_layers.append(X)
            elif key.startswith('dropout'):
                X = gc_layer(X)
        #print(X.shape, N.shape)
        #X = torch.stack(graph_layers, dim=0)
        #X = torch.mean(X, dim=0)
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

        self.gc_layers = nn.ModuleDict({})
        for i in range(n_glayer):
            self.gc_layers['gc{}'.format(i)] = GatedGraphConvolution(n_feat, n_feat, n_node, n_batch=n_batch)
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