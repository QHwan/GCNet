from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
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
