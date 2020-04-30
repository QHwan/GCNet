import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_feat)
        self.gc2 = GraphConvolution(n_feat, n_feat)
        self.nn1 = nn.Linear(n_feat, 8)
        self.nn2 = nn.Linear(8, 1)
        self.dropout = dropout

    def forward(self, X, A):
        X = F.relu(self.gc1(X, A))
        #x = F.dropout(x, self.dropout, training=self.training)
        X = self.gc2(X, A)

        X = torch.mean(X, dim=0)

        #print(x, x.shape)

        #x = torch.mean(x, dim=0)
        X = F.relu(self.nn1(X))
        X = self.nn2(X)
        #print(x); exit(1)
        return(X)