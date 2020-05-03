import random
import numpy as np
import torch
from torch.utils.data import Dataset

class FreeSolvDataset(Dataset):
    def __init__(self, npz_file, train_ratio=.8, val_ratio=.1, test_ratio=.1):
        self.file = np.load(npz_file, allow_pickle=True)
        self.Xs = self.file['Xs']
        self.As = self.file['As']
        self.Ys = self.file['Ys']

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
                self.Ys[idx].astype(np.float32)))