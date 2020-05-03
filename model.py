import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, dropout):
        super(GCN, self).__init__()

        n_gc_layers = 2
        self.gc_layers = nn.ModuleDict({})
        for i in range(n_gc_layers):
            self.gc_layers['gc{}'.format(i)] = GraphConvolution(n_feat, n_feat)
            self.gc_layers['relu{}'.format(i)] = nn.ReLU()
            
            if i == n_gc_layers-1:
                self.gc_layers['dropout{}'.format(i)] = nn.Dropout()


        self.nn_layers = nn.Sequential(
            nn.Linear(n_feat, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.dropout = dropout

    def forward(self, X, A):
        graph_layers = []
        graph_layers.append(X)

        for i, (key, gc_layer) in enumerate(self.gc_layers.items()):
            if key.startswith('gc'):
                X = gc_layer(X, A)
            else:
                X = gc_layer(X)
            graph_layers.append(X)
        
        X = torch.mean(torch.mean(torch.stack(graph_layers), dim=0), dim=0)

        X = self.nn_layers(X)
        return(X)