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

    def forward(self, x, adj, n):
        n_feature = x.shape[-1]
        #print(adj); exit(1)
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        #print(x, x.shape)

        '''
        x_pool = torch.zeros((len(n), n_feature))
        idx = 0
        #print(n)
        for i in range(len(n)):
            n_i = int(n[i].numpy())
            x_pool[i] = torch.sum(x[idx:idx+n_i], dim=0)
            idx += n_i

        x = torch.FloatTensor(x_pool)
        '''
        x = torch.mean(x, dim=0)

        #print(x, x.shape)

        #x = torch.mean(x, dim=0)
        x = F.relu(self.nn1(x))
        x = self.nn2(x)
        #print(x); exit(1)
        return(x)