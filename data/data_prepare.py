from __future__ import division
from __future__ import print_function

import random
import numpy as np
import sparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class GraphDataset(Dataset):
    def __init__(self, npz_file):
        self.file = np.load(npz_file, allow_pickle=True)
        self.Xs = self.file['Xs']
        self.As = self.file['As']
        self.Es = self.file['Es']
        self.Ns = self.file['Ns']
        self.Ys = self.file['Ys']
        self.max_n_nodes = self.file['max_n_nodes']

        # Y normalization
        if abs(np.max(self.Ys)) > abs(np.min(self.Ys)):
            self.norm_value = np.max(self.Ys)
        else:
            self.norm_value = np.min(self.Ys)

        self.Ys /= self.norm_value

    def __len__(self):
        return(len(self.Xs))

    def __getitem__(self, idx):
        return((torch.from_numpy(self.Xs[idx]).float(),
            torch.from_numpy(self.As[idx]).float(),
            self.Es[idx],
            self.Ns[idx],
            self.Ys[idx]))


def collate_data(dataset):
    batch_Xs = []
    batch_As = []
    batch_Es = []
    batch_Ns = []
    batch_Ys = []
    for i, (Xs, As, Es, Ns, Ys) in enumerate(dataset):
        batch_Xs.append(Xs)
        batch_As.append(As)
        batch_Es.append(Es)
        batch_Ns.append(Ns)
        batch_Ys.append(Ys)

    batch_Ns = np.array(batch_Ns)
    batch_Ys = np.array(batch_Ys)
    
    return(torch.stack(batch_Xs, dim=0),
        batch_As,
        batch_Es,
        torch.from_numpy(batch_Ns).float(),
        torch.from_numpy(batch_Ys).float())



def load_data(params):
    dataset = GraphDataset(npz_file=params['f'])
    n_data = len(dataset)
    n_train = int(n_data*params['train_ratio'])
    n_val = int(n_data*params['val_ratio'])
    n_test = n_data - n_train - n_val
    
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