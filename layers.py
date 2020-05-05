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
    def __init__(self, n_node, n_fea_in, n_fea_out, bias=True):
        super(GraphAttention, self).__init__()
        self.n_fea_in = n_fea_in
        self.n_fea_out = n_fea_out
        self.n_node = n_node
        self.W = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.a = Parameter(torch.FloatTensor(1, 2*n_fea_out))

        self.bias_W = Parameter(torch.FloatTensor(n_fea_out))
        self.bias_a = Parameter(torch.FloatTensor(2*n_fea_out))
        
        self.reset_parameters_uniform(self.W, self.bias_W)
        self.reset_parameters_uniform(self.a, self.bias_a)

    def reset_parameters_uniform(self, x, bias):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)

    def attention(self, XW, A, att):
        X1 = XW.unsqueeze(0)
        Y1 = XW.unsqueeze(1)
        X2 = X1.repeat(XW.shape[0], 1, 1)
        Y2 = Y1.repeat(1, XW.shape[0], 1)
        Z = torch.cat([X2, Y2], -1)
        Z1 = F.leaky_relu(torch.einsum("abc,cd->abd", (Z, self.a.T)).squeeze())
        denominator = torch.exp(Z1)
        numerator = torch.mm(A, denominator)

        for i in range(self.n_node):
            for j in range(self.n_node):
                if numerator[i,j] == 0:
                    continue
                att[i,j] = denominator[i,j]/numerator[i,j]

        #self.att = torch.div(denominator, numerator)
        #self.att[mask] = 0

    def forward(self, X, A):
        att = torch.FloatTensor(self.n_node, self.n_node)
 
        XW = torch.mm(X, self.W)
        self.attention(XW, A, att)
        buf = torch.mm(att, XW)
        H = torch.mm(A, buf)
        return(H)

    

