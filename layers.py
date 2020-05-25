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

    def reset_parameters_uniform(self, x):
        stdv = 1. / math.sqrt(x.size(0))
        x.data.uniform_(-stdv, stdv)
        #nn.init.xavier_uniform_(x, gain=1.414)


class GraphConvolution(CoreLayer):
    def __init__(self, n_fea_in, n_fea_out, n_node, n_batch):
        super().__init__(n_fea_in, n_fea_out, n_node, n_batch)
        self.W_self = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.W_nei = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W_self)
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias)

    def forward(self, H, A):
        H_filtered_self = torch.einsum("aij,jk->aik", (H, self.W_self)) 

        H_filtered_nei = torch.einsum("aij,jk->aik", (H, self.W_nei)) 
        H_filtered_nei = torch.bmm(A, H_filtered_nei) 
        H1 = H_filtered_self + H_filtered_nei + self.bias
        return(H1)


class MessagePassing(CoreLayer):
    def __init__(self, n_node_fea, n_edge_fea, n_node, n_batch):
        super().__init__(n_node_fea, n_edge_fea, n_node, n_batch)
        self.W_nei = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.bias = Parameter(torch.FloatTensor(n_node_fea))
        
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias)

    def concat_H_E(self, H, E):
        H2 = H.unsqueeze(2).repeat(1, 1, self.n_node, 1)
        H1 = H.unsqueeze(1).repeat(1, self.n_node, 1, 1)
        return(torch.cat((H2, H1, E), 3))

    def forward(self, H, A, E, N):
        HE = self.concat_H_E(H, E)
        HE = A.unsqueeze(3) * HE      
        HE = torch.sum(HE, dim=2)

        H_filtered_nei = torch.einsum("aij,jk->aik", (HE, self.W_nei)) 
        H_filtered_nei = torch.bmm(A, H_filtered_nei) 
        H1 = H_filtered_nei + self.bias
        return(H1)


class AttentionMessagePassing(CoreLayer):
    def __init__(self, n_node_fea, n_edge_fea, n_node, n_batch):
        super().__init__(n_node_fea, n_edge_fea, n_node, n_batch)
        self.W_att = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.W_nei = Parameter(torch.FloatTensor(2*n_node_fea+n_edge_fea, n_node_fea))
        self.bias_att = Parameter(torch.FloatTensor(n_node_fea))
        self.bias_nei = Parameter(torch.FloatTensor(n_node_fea))
        
        self.reset_parameters_uniform(self.W_att)
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias_att)
        self.reset_parameters_uniform(self.bias_nei)

    def concat_H_E(self, H, E):
        H2 = H.unsqueeze(2).repeat(1, 1, self.n_node, 1)
        H1 = H.unsqueeze(1).repeat(1, self.n_node, 1, 1)
        return(torch.cat((H2, H1, E), 3))

    def forward(self, H, A, E, N):
        HE = self.concat_H_E(H, E)
        HE = A.unsqueeze(3) * HE
        HE_att = torch.einsum("aijk,kl->aijl", (HE, self.W_att)) + self.bias_att
        HE_conv = torch.einsum("aijk,kl->aijl", (HE, self.W_nei)) + self.bias_nei
        HE1 = F.sigmoid(HE_att) * F.relu(HE_conv)
        HE1 = torch.sum(HE1, dim=1)
        return(HE1)



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
