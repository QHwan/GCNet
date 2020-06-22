from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class CoreLayer(nn.Module):
    def __init__(self, n_fea_in, n_fea_out):
        super(CoreLayer, self).__init__()
        self.n_fea_in = n_fea_in
        self.n_fea_out = n_fea_out

    def reset_parameters_uniform(self, x):
        stdv = 1. / math.sqrt(x.size(0))
        x.data.uniform_(-stdv, stdv)
        #nn.init.xavier_uniform_(x, gain=1.414)


class GraphConvolution(CoreLayer):
    def __init__(self, n_fea_in, n_fea_out):
        super().__init__(n_fea_in, n_fea_out)
        self.W_self = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.W_nei = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W_self)
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias)

    def forward(self, H, A):
        H_filtered_self = torch.mm(H, self.W_self) 

        H_filtered_nei = torch.mm(H, self.W_nei) 
        H_filtered_nei = torch.mm(A, H_filtered_nei) 
        H1 = H_filtered_self + H_filtered_nei + self.bias
        return(H1)


class MessageGraphConvolution(CoreLayer):
    def __init__(self, n_node_fea, n_edge_fea):
        super().__init__(n_node_fea, n_edge_fea)
        self.W_self = Parameter(torch.FloatTensor(n_node_fea, n_node_fea))
        self.W_nei = Parameter(torch.FloatTensor(n_node_fea + n_edge_fea, n_node_fea))
        self.bias = Parameter(torch.FloatTensor(n_node_fea))
        
        self.reset_parameters_uniform(self.W_self)
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias)

    def forward(self, H, A, E):
        H_filtered_self = torch.mm(H, self.W_self) 

        H_filtered_nei = torch.cat((torch.mm(A, H), E), dim=1)

        H_filtered_nei = torch.mm(H_filtered_nei, self.W_nei) 
        H1 = H_filtered_self + H_filtered_nei + self.bias
        return(H1)


class MessageGraphAttention(CoreLayer):
    def __init__(self, n_node_fea, n_edge_fea):
        super().__init__(n_node_fea, n_edge_fea)
        self.W = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.W_att = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.bias = Parameter(torch.FloatTensor(n_node_fea))
        
        self.reset_parameters_uniform(self.W)
        self.reset_parameters_uniform(self.W_att)
        self.reset_parameters_uniform(self.bias)

        self.n_node_fea = n_node_fea

    def _self_cat(self, H):
        H1 = H.unsqueeze(0).repeat(H.size()[0], 1, 1)
        H2 = H.unsqueeze(1).repeat(1, H.size()[0], 1)
        return(torch.cat([H2, H1], dim=2))

    def forward(self, H, A, E):
        H_self_cat = self._self_cat(H)
        HE = torch.cat([H_self_cat, E], dim=2) 
        HE = HE * A.unsqueeze(2)

        H = torch.einsum("abi,ij->abj", (HE, self.W)) + self.bias
        H_att = torch.einsum("abi,ij->abj", (HE, self.W_att)) + self.bias

        _H = F.relu(H) * torch.sigmoid(H_att)
        _H = torch.sum(_H, dim=1)

        return(_H)

