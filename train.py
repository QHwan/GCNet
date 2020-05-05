from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import GCN, GAT
from data.data_prepare import FreeSolvDataset


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
parser.add_argument('--model', type=str, default='GAT',
                    help='Network Model: GCN, GAT')
parser.add_argument('--train_ratio', type=float, default=.8)
parser.add_argument('--val_ratio', type=float, default=.1)
parser.add_argument('--n_fold', type=int, default=5,
                    help='number of repetitive training for error valuation')

args = parser.parse_args()


def train(epoch,
          train_loader,
          val_loader,
          test_loader,
          n_node,
          n_feat,
          model=args.model,
          optimizer='Adam',
          loss='mse',
          n_hid=args.hidden,
          dropout=args.dropout,
          lr=args.lr):

    if model.lower() == 'gcn':
        model = GCN(n_feat=n_feat,
                    n_hid=n_hid,
                    dropout=dropout)

    if model.lower() == 'gat':
        model = GAT(n_node=n_node,
                    n_feat=n_feat,
                    n_hid=n_hid,
                    dropout=dropout)

    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss.lower() == 'mse':
        loss_fn = nn.MSELoss()
    
    n_train = len(train_loader)
    n_val = len(val_loader)

    outputs_train = []
    outputs_val = []


    for i in range(epoch):
        t = time.time()
        running_loss_train = 0
        running_loss_val = 0

        model.train()
        for j, batch in enumerate(train_loader):
            X_batch, A_batch, Y_batch = batch
            optimizer.zero_grad()

            output_train = model(X_batch[0], A_batch[0])
            loss_train = loss_fn(output_train.T, Y_batch)

            loss_train.backward()
            optimizer.step()
            running_loss_train += loss_train.data

            if i == epoch-1:
                for output, Y in zip(output_train, Y_batch):
                    outputs_train.append([output.detach().numpy(),
                                          Y.detach().numpy()])


        model.eval()
        for j, batch in enumerate(val_loader):
            X_batch, A_batch, Y_batch = batch
            output_val = model(X_batch[0], A_batch[0])
            loss_val = loss_fn(output_val.T, Y_batch)

            running_loss_val += loss_val.data

            if i == epoch-1:
                for output, Y in zip(output_val, Y_batch):
                    outputs_val.append([output.detach().numpy(),
                                        Y.detach().numpy()])

        print('Epoch: {:04d}'.format(i+1),
            'loss_train: {:.4f}'.format(running_loss_train/n_train),
            'loss_val: {:.4f}'.format(running_loss_val/n_val),
            'time: {:.4f}s'.format(time.time() - t))


    return(np.array(outputs_train), np.array(outputs_val))


def load_data(npz_file, train_ratio=args.train_ratio, val_ratio=args.val_ratio):
    dataset = FreeSolvDataset(npz_file=npz_file)
    n_data = len(dataset)
    n_train = int(n_data*train_ratio)
    n_val = int(n_data*val_ratio)
    n_test = n_data - n_train - n_val
    trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])

    dataloader_options = {'batch_size': 1,
                        'shuffle': True,
                        'pin_memory': False}

    train_loader = DataLoader(trainset, **dataloader_options)
    val_loader = DataLoader(valset, **dataloader_options)
    test_loader = DataLoader(testset, **dataloader_options)

    return(dataset, train_loader, val_loader, test_loader)


def cal_loss(x, kind):
    if kind.lower() == 'rmse':
        return(mean_squared_error(x[:,0], x[:,1], squared=False))

# Train model
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

t_total = time.time()

n_fold = args.n_fold
losses_train = np.zeros(n_fold)
losses_val = np.zeros(n_fold)
for i in range(n_fold):
    dataset, train_loader, val_loader, test_loader = load_data(npz_file='./data/dataset/freesolv.npz')
    n_node, n_feat = dataset[0][0].shape
    outputs_train, outputs_val = train(args.epochs, train_loader, val_loader, test_loader,
                                    n_feat=n_feat, n_node=n_node)
    outputs_train *= dataset.norm_value
    outputs_val *= dataset.norm_value

    losses_train[i] = cal_loss(outputs_train, kind='rmse')
    losses_val[i] = cal_loss(outputs_val, kind='rmse')

print('Train error: {} +- {}'.format(np.mean(losses_train), np.std(losses_train)))
print('Validation error: {} +- {}'.format(np.mean(losses_val), np.std(losses_val)))

x = np.linspace(-0.2*dataset.norm_value, dataset.norm_value, 100)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(outputs_val[:,0], outputs_val[:,1], s=5, alpha=.5)
ax.plot(x, x)
ax.set_xlabel('Prediction (kcal/mol)')
ax.set_ylabel('Experiment (kcal/mol)')
plt.show()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

