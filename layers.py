from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.W = Parameter(torch.FloatTensor(n_in, n_out))

        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters_uniform()

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):
        #buf = torch.mm(X, self.W)
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
        self.a = Parameter(torch.FloatTensor(1, 2*n_fea_out))

        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))
        self.bias_a = Parameter(torch.FloatTensor(n_node))
        
        self.reset_parameters_uniform(self.W, self.bias_W)
        self.reset_parameters_uniform(self.a, self.bias_a)

    def reset_parameters_uniform(self, x, bias):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)

    def attention(self, XW, A, att):
        X1 = XW.unsqueeze(1)
        Y1 = XW.unsqueeze(2)
        X2 = X1.repeat(1, XW.shape[1], 1, 1)
        Y2 = Y1.repeat(1, 1, XW.shape[1], 1)
        Z = torch.cat([X2, Y2], 3)
        Z1 = torch.einsum("abcd,de->abce", (Z, self.a.T)).squeeze() + self.bias_a
        denominator = torch.exp(Z1)
        numerator = torch.bmm(A, denominator)

        for i in range(self.n_batch):
            mask = numerator[i] != 0

            buf = torch.div(denominator[i][mask], numerator[i][mask])
            att[i][mask] = buf


    def forward(self, X, A):
        att = torch.FloatTensor(self.n_batch, self.n_node, self.n_node)
 
        XW = torch.einsum("abc,cd->abd", (X, self.W))
        self.attention(XW, A, att)
        buf = torch.bmm(att, XW)
        H = torch.bmm(A, buf) + self.bias_W
        return(H)

    

