from __future__ import print_function, division

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from scipy.linalg import block_diag


def get_train_val_test_loader(dataset, collate_fn,
                              batch_size=64, train_ratio=.8,
                              val_ratio=.1, test_ratio=.1,
                              num_workers=1, pin_memory=False):
    total_size = len(dataset)

    indices = list(range(total_size))

    train_size = int(train_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              pin_memory=pin_memory)

    return(train_loader)


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch

    Parameters
    ----------
    dataset_list: list of tuples for each data point.
        (atom_fea, adj, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        adj: torch.Tensor shape (n_i, n_i)
        target: torch.Tensor shape (1, )

    Returns
    -------
    N = sum(n_i)

    batch_atom_fea: torch.Tensor shape (N, atom_fea_len)
    batch_adj: torch.Tensor shape (N, N)
    batch_target: torch.Tensor shape (N, 1)
    """
    batch_atom_fea, batch_adj = [], []
    batch_target = []

    for i, (atom_fea, adj, target) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]

        batch_atom_fea.append(atom_fea)
        batch_adj.append(adj.numpy())
        batch_target.append(target)

    return(torch.cat(batch_atom_fea, dim=0),
           torch.from_numpy(block_diag(*batch_adj)),
           torch.stack(batch_target, dim=0))