from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *

def make_nn_layers(n_hid, n_feat, dropout):
    nn_module = []
    nn_module.append(nn.Linear(n_feat, n_hid))
    nn_module.append(nn.ReLU())
    nn_module.append(nn.Dropout(dropout))
    nn_module.append(nn.Linear(n_hid, 1))
    return(nn.Sequential(*nn_module))


class CoreModule(nn.Module):
    def __init__(self, params):
        super(CoreModule, self).__init__()
        self.n_node_fea = params['n_node_fea']
        self.n_edge_fea = params['n_edge_fea']
        self.n_hid = params['n_hid']
        self.n_glayer = params['n_glayer']
        self.n_batch = params['n_batch']
        self.dropout = params['dropout']
        self.n_embed_fea = 64

        self.nn_layers = make_nn_layers(self.n_hid,
                                        self.n_embed_fea,
                                        self.dropout)

class GCN(CoreModule):
    def __init__(self, params):
        super().__init__(params)

        self.embed_layer = nn.Linear(self.n_node_fea, self.n_embed_fea)

        self.gc1 = GraphConvolution(self.n_embed_fea, self.n_embed_fea)
        self.gc2 = GraphConvolution(self.n_embed_fea, self.n_embed_fea)
        self.bn1 = nn.BatchNorm1d(self.n_embed_fea)        

    def forward(self, X, A, E, E_avg, N):
        X = self.embed_layer(X)

        X = X + F.leaky_relu(self.gc1(X, A))
        X = X + F.leaky_relu(self.gc2(X, A))
        X = self.bn1(X)
        
        X = self.pooling(X, N) 

        X = self.nn_layers(X)
        return(X)

    def pooling(self, X, N):
        summed_X = [torch.mean(X[n], dim=0, keepdim=True)
                    for n in N]
        return(torch.cat(summed_X, dim=0))



class MGCN(CoreModule):
    def __init__(self, params):
        super().__init__(params)

        self.embed_layer = nn.Linear(self.n_node_fea, self.n_embed_fea)

        self.gc1 = MessageGraphConvolution(self.n_embed_fea, self.n_edge_fea)
        self.gc2 = MessageGraphConvolution(self.n_embed_fea, self.n_edge_fea)
        self.bn1 = nn.BatchNorm1d(self.n_embed_fea)        

    def forward(self, X, A, E, E_avg, N):
        X = self.embed_layer(X)

        X = X + F.leaky_relu(self.gc1(X, A, E_avg))
        X = X + F.leaky_relu(self.gc2(X, A, E_avg))
        X = self.bn1(X)
        
        X = self.pooling(X, N) 

        X = self.nn_layers(X)
        return(X)

    def pooling(self, X, N):
        summed_X = [torch.mean(X[n], dim=0, keepdim=True)
                    for n in N]
        return(torch.cat(summed_X, dim=0))


class MGAT(CoreModule):
    def __init__(self, params):
        super().__init__(params)

        self.embed_layer = nn.Linear(self.n_node_fea, self.n_embed_fea)

        self.gc1 = MessageGraphAttention(self.n_embed_fea, self.n_edge_fea)
        self.gc2 = MessageGraphAttention(self.n_embed_fea, self.n_edge_fea)
        self.bn1 = nn.BatchNorm1d(self.n_embed_fea)        

    def forward(self, X, A, E, E_avg, N):
        X = self.embed_layer(X)

        # activation is included in layers
        X = X + self.gc1(X, A, E)
        X = X + self.gc2(X, A, E)
        #X = self.bn1(X)
        
        X = self.pooling(X, N) 

        X = self.nn_layers(X)
        return(X)

    def pooling(self, X, N):
        summed_X = [torch.mean(X[n], dim=0, keepdim=True)
                    for n in N]
        return(torch.cat(summed_X, dim=0))

