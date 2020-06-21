from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class GraphDataset(Dataset):
    def __init__(self, npz_file):
        self.file = np.load(npz_file, allow_pickle=True)
        self.Xs = self.file['Xs']
        self.As = self.file['As']
        self.Es_idx = self.file['Es_idx']
        self.Es_fea = self.file['Es_fea']
        self.Ns = self.file['Ns']
        self.Ys = self.file['Ys']

        self.Ys /= self.Ys.min()

        self.n_node_fea = self.Xs[0].shape[1]
        self.n_edge_fea = len(self.Es_fea[0][0])
        self.n_out_fea = len(self.Ys[0])

    def __len__(self):
        return(len(self.Xs))

    def __getitem__(self, idx):
        return(
            (
                self.Xs[idx],
                self.As[idx],
                self.Es_idx[idx],
                self.Es_fea[idx],
                self.Ns[idx],
                self.Ys[idx],
            )
        )

if __name__ == "__main__":
    dataset = GraphDataset('dataset/freesolv.npz')
    print(dataset[0])
    exit(1)


def collate_data(dataset):
    batch_Xs = []
    batch_Ns = []
    batch_Ys = []

    # first make X, Y batch
    N_tot = 0
    for i, (X, A, _, _, N, Y) in enumerate(dataset):
        for x in X:
            batch_Xs.append(x)
        batch_Ys.append(Y)
        N_tot += N

    # second make A, N batch
    batch_As = np.zeros((N_tot, N_tot))
    start_idx = 0
    for i, (_, A, _, _, N, Y) in enumerate(dataset):
        batch_As[start_idx:start_idx+N, start_idx:start_idx+N] += A
        
        idx_atoms = torch.from_numpy(np.array(range(start_idx, start_idx+N))).long()
        batch_Ns.append(idx_atoms)

        start_idx += N

    batch_Xs = np.array(batch_Xs)
    batch_As = np.array(batch_As)
    batch_Ys = np.array(batch_Ys)
    
    return(torch.from_numpy(batch_Xs).float(),
        torch.from_numpy(batch_As).float(),
        batch_Ns,
        torch.from_numpy(batch_Ys).float())



def load_data(params):
    dataset = GraphDataset(npz_file=params['f'])
    n_data = len(dataset)
    n_train = int(n_data*params['train_ratio'])
    n_val = int(n_data*params['val_ratio'])
    n_test = n_data - n_train - n_val

    params['n_train'] = n_train
    params['n_val'] = n_val
    params['n_test'] = n_test
    
    #trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])
    trainset = Subset(dataset, list(range(n_train)))
    valset = Subset(dataset, list(range(n_train, n_train+n_val)))
    testset = Subset(dataset, list(range(n_train+n_val, n_train+n_val+n_test)))

    dataloader_args = {'batch_size': params['n_batch'],
                       'shuffle': True,
                       'pin_memory': False,
                       'drop_last': False,
                       'collate_fn': collate_data,}

    train_loader = DataLoader(trainset, **dataloader_args)
    val_loader = DataLoader(valset, **dataloader_args)
    test_loader = DataLoader(testset, **dataloader_args)

    return(dataset, train_loader, val_loader, test_loader)