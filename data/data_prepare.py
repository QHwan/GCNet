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

        self.norm = self.Ys.min()
        self.Ys /= self.norm

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



def collate_data(dataset):
    batch_Xs = []
    batch_Es = []
    batch_Es_avg = []
    batch_Ns = []
    batch_Ys = []


    # first make X, Y batch, N_tot, n_edge_fea
    N_tot = 0
    for i, (X, A, _, E_fea, N, Y) in enumerate(dataset):
        for x in X:
            batch_Xs.append(x)
        batch_Ys.append(Y)
        N_tot += N

        if i == 0:
            n_edge_fea = np.shape(E_fea)[1]


<<<<<<< HEAD

    # second make A, E_idx, E_fea, N batch
    batch_As = np.zeros((N_tot, N_tot))
=======
    # second make A, E_idx, E_fea, N batch
    batch_As = np.zeros((N_tot, N_tot))
    batch_As = np.eye(N_tot)
>>>>>>> 7f04a9070527c4e52040ad48cad2559b7ba95769
    batch_Es = np.zeros((N_tot, N_tot, n_edge_fea))
    batch_Es_avg = np.zeros((N_tot, n_edge_fea))
    start_idx = 0
    for i, (_, A, E_idx, E_fea, N, Y) in enumerate(dataset):
        batch_As[start_idx:start_idx+N, start_idx:start_idx+N] += A

<<<<<<< HEAD
        '''
=======
>>>>>>> 7f04a9070527c4e52040ad48cad2559b7ba95769
        for j, e_idx in enumerate(E_idx):
            #batch_Es_idx.append([e_idx[0] + start_idx, e_idx[1] + start_idx])
            batch_Es[e_idx[0] + start_idx, e_idx[1] + start_idx] += E_fea[j]
            batch_Es_avg[e_idx[0] + start_idx] += E_fea[j]
<<<<<<< HEAD
        '''
        E_idx = np.array(E_idx)
        batch_Es[E_idx[:,0] + start_idx, E_idx[:,1] + start_idx] += E_fea
        batch_Es_avg[E_idx[:,0] + start_idx] += E_fea
=======
>>>>>>> 7f04a9070527c4e52040ad48cad2559b7ba95769
        
        idx_atoms = torch.from_numpy(np.array(range(start_idx, start_idx+N))).long()
        batch_Ns.append(idx_atoms)

        start_idx += N


    batch_Xs = np.array(batch_Xs)
    batch_As = np.array(batch_As)
    batch_Es = np.array(batch_Es)
    batch_Es_avg = np.array(batch_Es_avg)
    batch_Ys = np.array(batch_Ys)

<<<<<<< HEAD

=======
>>>>>>> 7f04a9070527c4e52040ad48cad2559b7ba95769
    
    return(torch.from_numpy(batch_Xs).float(),
        torch.from_numpy(batch_As).float(),
        torch.from_numpy(batch_Es).float(),
        torch.from_numpy(batch_Es_avg).float(),
        batch_Ns,
        torch.from_numpy(batch_Ys).float())



def load_data(params):
    dataset = GraphDataset(npz_file=params['f'])
    n_data = len(dataset)
    n_train = int(n_data*params['train_ratio'])
    n_val = int(n_data*params['val_ratio'])
    n_test = n_data - n_train - n_val

    print(n_data, n_train, n_val, n_test)

    params['n_train'] = n_train
    params['n_val'] = n_val
    params['n_test'] = n_test
    
    #trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])
    trainset = Subset(dataset, list(range(n_train)))
    valset = Subset(dataset, list(range(n_train, n_train+n_val)))
    testset = Subset(dataset, list(range(n_train+n_val, n_train+n_val+n_test)))

    dataloader_args = {'batch_size': params['n_batch'],
                       'shuffle': False,
                       'pin_memory': True,
                       'drop_last': False,
                       'collate_fn': collate_data,
                       'num_workers': 4,}

    train_loader = DataLoader(trainset, **dataloader_args)
    val_loader = DataLoader(valset, **dataloader_args)
    test_loader = DataLoader(testset, **dataloader_args)

    return(dataset, train_loader, val_loader, test_loader)
