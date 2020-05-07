from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.W = Parameter(torch.FloatTensor(n_in, n_out))
        self.bias_W = Parameter(torch.FloatTensor(n_out))
        
        self.reset_parameters_uniform(self.W, self.bias_W)

    def reset_parameters_uniform(self, x, bias):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):
        buf = torch.einsum("abc,cd->abd", (X, self.W))
        H = torch.bmm(A, buf)

        if self.bias is not None:
            return(H + self.bias)
        else:
            return(H)


class GraphAttention(nn.Module):
    def __init__(self, n_node, n_fea_in, n_fea_out, n_batch, bias=True):
        super(GraphAttention, self).__init__()
        self.n_fea_in = n_fea_in
        self.n_fea_out = n_fea_out
        self.n_node = n_node
        self.n_batch = n_batch
        self.W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.a = Parameter(torch.FloatTensor(n_fea_out, n_fea_out))

        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))
        self.bias_a = Parameter(torch.FloatTensor(n_node))
        
        self.reset_parameters_uniform(self.W, self.bias_W)
        self.reset_parameters_uniform(self.a, self.bias_a)

    def reset_parameters_uniform(self, x, bias):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)

    def concat_ij(self, x):
        x1 = x.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        y1 = x.unsqueeze(2).repeat(1, 1, x.shape[1], 1)
        return(torch.cat([x1, y1], 3))

    def attention(self, XW, A):
        XWT = XW.permute(0, 2, 1)
        buf = torch.einsum("ij,ajk->ajk", (self.a, XWT))
        Z = torch.einsum("aij,jk->ajk", (XW, self.a))
        return(F.relu(Z))


    def forward(self, X, A): 
        XW = torch.einsum("abc,cd->abd", (X, self.W))
        #self.attention(XW, A, att)
        att = self.attention(X, A)
        buf = torch.bmm(XW, att)
        H = torch.bmm(A, buf) + self.bias_W
        return(H)

    

