from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class CoreLayer(nn.Module):
    def __init__(self, n_fea_in, n_fea_out, n_node, n_batch):
        super(CoreLayer, self).__init__()
        self.n_fea_in = n_fea_in
        self.n_fea_out = n_fea_out
        self.n_node = n_node
        self.n_batch = n_batch

    def reset_parameters_uniform(self, x, bias):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)


class GraphConvolution(CoreLayer):
    def __init__(self, n_fea_in, n_fea_out, n_node, n_batch):
        super().__init__(n_fea_in, n_fea_out, n_node, n_batch)
        self.W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W, self.bias_W)

    def forward(self, X, A):
        buf = torch.einsum("abc,cd->abd", (X, self.W))
        H = torch.bmm(A, buf) + self.bias_W
        return(H)


class GatedGraphConvolution(CoreLayer):
    def __init__(self, n_fea_in, n_fea_out, n_node, n_batch):
        super().__init__(n_fea_in, n_fea_out, n_node, n_batch)
        self.W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))

        self.self_W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias_self_W = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W, self.bias_W)
        self.reset_parameters_uniform(self.self_W, self.bias_self_W)

    def forward(self, X, A):
        buf = torch.einsum("abc,cd->abd", (X, self.W))
        H = torch.bmm(A, buf) + self.bias_W
        H += torch.einsum("abc,cd->abd", (X, self.self_W))
        return(H)



class GraphAttention(CoreLayer):
    def __init__(self, n_node, n_fea_in, n_fea_out, n_batch):
        super().__init__(n_fea_in, n_fea_out, n_node, n_batch)
        self.W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.att_W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.att_W2 = Parameter(torch.FloatTensor(n_node, n_node))
        self.a = Parameter(torch.FloatTensor(n_fea_out, n_fea_out))

        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))
        self.bias_a = Parameter(torch.FloatTensor(n_node))

        self.self_W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias_self_W = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W, self.bias_W)
        self.reset_parameters_uniform(self.att_W, self.bias_W)
        self.reset_parameters_uniform(self.att_W2, self.bias_W)

        self.reset_parameters_uniform(self.a, self.bias_a)
        self.reset_parameters_uniform(self.self_W, self.bias_self_W)

    def concat_ij(self, x):
        x1 = x.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        y1 = x.unsqueeze(2).repeat(1, 1, x.shape[1], 1)
        return(torch.cat([x1, y1], 3))

    def attention(self, XW):
        XWT = XW.permute(0, 2, 1)
        buf = torch.einsum("ij,ajk->ajk", (self.a, XWT))
        Z = torch.einsum("aij,ajk->aik", (XW, buf))
        #Z = torch.einsum("aij,ajk->aik", (XW, XWT))
        
        #Z = torch.einsum("aij,jk->aik", (Z, self.att_W2))
        for i in range(len(Z)):
            Z[i].fill_diagonal_(0)
        return(F.leaky_relu(Z))

    def forward(self, X, A): 
        att_X = torch.einsum("abc,cd->abd", (X, self.att_W))
        att = self.attention(att_X)

        buf = torch.einsum("aij,jk->aik", (X, self.W))
        H = torch.einsum("aij,ajk->aik", (att, buf)) + self.bias_W
        #H = torch.bmm(A, H)
        H += torch.einsum("abc,cd->abd", (X, self.self_W))
        return(H)
