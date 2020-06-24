from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def reset_parameters_uniform(x):
    stdv = 1. / math.sqrt(x.size(0))
    x.data.uniform_(-stdv, stdv)


class GraphConvolution(nn.Module):
    def __init__(self, n_node_fea):
        super(GraphConvolution, self).__init__()

        self.W = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.W_conv = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.W_pool1 = Parameter(torch.FloatTensor(n_node_fea*2, n_node_fea*2))
        self.W_pool2 = Parameter(torch.FloatTensor(n_node_fea*2, n_node_fea))
        self.bias1 = Parameter(torch.FloatTensor(n_node_fea*2))
        self.bias2 = Parameter(torch.FloatTensor(n_node_fea))
        
        reset_parameters_uniform(self.W)
        reset_parameters_uniform(self.W_conv)
        reset_parameters_uniform(self.W_pool1)
        reset_parameters_uniform(self.W_pool2)
        reset_parameters_uniform(self.bias1)
        reset_parameters_uniform(self.bias2)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, H, A):
        _H = torch.mm(H, self.W)

        _H_conv = torch.mm(H, self.W_conv) 
        _H_conv = torch.mm(A, _H_conv) 

        _H = torch.cat([_H, _H_conv], dim=1)
        _H = F.relu(_H)

        _H = torch.mm(_H, self.W_pool1) + self.bias1
        _H = F.relu(_H)
        _H = self.drop1(_H)

        _H = torch.mm(_H, self.W_pool2) + self.bias2
        _H = F.relu(_H)
        _H = self.drop2(_H)
        return(_H)


class MessageGraphConvolution(nn.Module):
    def __init__(self, n_node_fea, n_edge_fea):
        super(MessageGraphConvolution, self).__init__()
        self.W = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.W_conv = Parameter(torch.FloatTensor(n_node_fea + n_edge_fea, n_node_fea))
        self.W_pool1 = Parameter(torch.FloatTensor(n_node_fea*2, n_node_fea*2))
        self.W_pool2 = Parameter(torch.FloatTensor(n_node_fea*2, n_node_fea))
        self.bias1 = Parameter(torch.FloatTensor(n_node_fea*2))
        self.bias2 = Parameter(torch.FloatTensor(n_node_fea))
        
        reset_parameters_uniform(self.W)
        reset_parameters_uniform(self.W_conv)
        reset_parameters_uniform(self.W_pool1)
        reset_parameters_uniform(self.W_pool2)
        reset_parameters_uniform(self.bias1)
        reset_parameters_uniform(self.bias2)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
       
    def forward(self, H, A, E):
        _H = torch.mm(H, self.W)

        _H_conv = torch.cat([torch.mm(A, H), E], dim=1)
        _H_conv = torch.mm(_H_conv, self.W_conv) 

        _H = torch.cat([_H, _H_conv], dim=1)
        _H = F.relu(_H)

        _H = torch.mm(_H, self.W_pool1) + self.bias1
        _H = F.relu(_H)
        _H = self.drop1(_H)

        _H = torch.mm(_H, self.W_pool2) + self.bias2
        _H = F.relu(_H)
        _H = self.drop2(_H)

        return(_H)


class MessageGraphAttention(nn.Module):
    def __init__(self, n_node_fea, n_edge_fea):
        super(MessageGraphAttention, self).__init__()
        self.W = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.W_att = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.bias1 = Parameter(torch.FloatTensor(n_node_fea))
        self.bias2 = Parameter(torch.FloatTensor(n_node_fea))

        self.W_pool1 = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.W_pool2 = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.bias3 = Parameter(torch.FloatTensor(n_node_fea))
        self.bias4 = Parameter(torch.FloatTensor(n_node_fea))
       
        reset_parameters_uniform(self.W)
        reset_parameters_uniform(self.W_att)
        reset_parameters_uniform(self.W_pool1)
        reset_parameters_uniform(self.W_pool2)
        reset_parameters_uniform(self.bias1)
        reset_parameters_uniform(self.bias2)
        reset_parameters_uniform(self.bias3)
        reset_parameters_uniform(self.bias4)

        self.n_node_fea = n_node_fea

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

    def _expand(self, H):
        H1 = H.unsqueeze(1).repeat(1, H.size()[0], 1)
        H2 = H.unsqueeze(0).repeat(H.size()[0], 1, 1)
        return(torch.cat([H1, H2], dim=2))

    def forward(self, H, A, E):
        _H_expand = self._expand(H)
        _HE = torch.cat([_H_expand, E], dim=2) 
        _HE = _HE * A.unsqueeze(2)

        _H = torch.einsum("abi,ij->abj", (_HE, self.W)) + self.bias1
        _H_att = torch.einsum("abi,ij->abj", (_HE, self.W_att)) + self.bias2

        _H = F.relu(_H) * torch.sigmoid(_H_att)
        _H = torch.sum(_H, dim=1)

        _H = torch.mm(_H, self.W_pool1) + self.bias3
        _H = F.relu(_H)
        _H = self.drop1(_H)

        _H = torch.mm(_H, self.W_pool2) + self.bias4
        _H = F.relu(_H)
        _H = self.drop1(_H)

        return(_H)

