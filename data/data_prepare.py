from __future__ import division
from __future__ import print_function

import random
import numpy as np
import sparse
from torch.utils.data import Dataset

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
        return((self.Xs[idx].astype(np.float32),
                self.As[idx].astype(np.float32),
                self.Es[idx].astype(np.float32),
                self.Ns[idx].astype(np.float32),
                self.Ys[idx].astype(np.float32)))