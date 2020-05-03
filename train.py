from __future__ import division
from __future__ import print_function

import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import GCN
from data.data_prepare import data_prepare


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default='GCN',
                    help='Network Model: GCN')
parser.add_argument('--o', type=str, default=None,
                    help='Save model')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')

args = parser.parse_args()


np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))


# Load data
dataset, _ = data_prepare(filename='./data/dataset/freesolv.npz')
Xs, As, Ys = dataset
Xs_train, Xs_val, Xs_test = Xs
As_train, As_val, As_test = As
Ys_train, Ys_val, Ys_test = Ys

# Model and optimizer
n_feat = Xs_train[0].shape[-1]
model = GCN(n_feat=n_feat,
            n_hid=args.hidden,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_fn = torch.nn.MSELoss()

n_batch = 1

def train(epoch, Xs, As, Ys):
    Xs_train, Xs_val, _ = Xs
    As_train, As_val, _ = As
    Ys_train, Ys_val, _ = Ys
    
    n_train = len(Xs_train)
    n_val = len(Xs_val)

    outputs_train = []
    outputs_val = []


    for i in range(epoch):
        t = time.time()
        running_loss_train = 0
        running_loss_val = 0

        model.train()
        for j in range(n_train):
            optimizer.zero_grad()

            output_train = model(torch.FloatTensor(Xs_train[j]),
                                 torch.FloatTensor(As_train[j]))
            loss_train = loss_fn(output_train,
                                 torch.FloatTensor([Ys_train[j]]))
            loss_train.backward()
            optimizer.step()
            running_loss_train += loss_train.data

            if i == epoch-1:
                outputs_train.append(*output_train.detach().numpy())


        model.eval()
        for j in range(n_val):
            output_val = model(torch.FloatTensor(Xs_val[j]),
                               torch.FloatTensor(As_val[j]))
            loss_val = loss_fn(output_val,
                               torch.FloatTensor([Ys_val[j]]))

            running_loss_val += loss_val.data

            if i == epoch-1:
                outputs_val.append(*output_val.detach().numpy())

        print('Epoch: {:04d}'.format(i+1),
            'loss_train: {:.4f}'.format(running_loss_train/n_train),
            'loss_val: {:.4f}'.format(running_loss_val/n_val),
            'time: {:.4f}s'.format(time.time() - t))


    return(outputs_train, outputs_val)




# Train model
t_total = time.time()
outputs_train, outputs_val = train(args.epochs, Xs, As, Ys)

x = np.linspace(-1, 1, 100)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(Ys_train, outputs_train, s=5, alpha=.5)
ax[0].plot(x, x)
ax[1].scatter(Ys_val, outputs_val, s=5, alpha=.5)
ax[1].plot(x, x)
plt.show()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Save File
if args.o is not None:
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, args.o)
