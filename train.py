from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from model import *
from data.data_prepare import GraphDataset, load_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str)
parser.add_argument('--best_model', type=str, default='pretrained/best_model.pth.tar')
parser.add_argument('--resume', type=str)
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--n_glayer', type=int, default=1,
                    help='Number of graph layer')
parser.add_argument('--n_nlayer', type=int, default=1,
                    help='Number of dense layer')
parser.add_argument('--n_hid', nargs='+', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default='GCN',
                    help='Network Model: GCN, GAT, GCN_Gate')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--loss_fn', type=str, default='mse')
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--train_ratio', type=float, default=.6)
parser.add_argument('--val_ratio', type=float, default=.2)
parser.add_argument('--n_batch', type=int, default=8,
                    help='number of mini-batch')

args = parser.parse_args()
params = vars(args)


def train(train_loader,
          val_loader,
          test_loader,
          params):

    if params['model'].lower() == 'gcn':
        model = GCN(params)

    elif params['model'].lower() == 'gat':
        model = GAT(params)

    elif params['model'].lower() == 'gcn_gate':
        model = GCN_Gate(params)

    elif params['model'].lower() == 'mpnn':
        model = MPNN(params)

    elif params['model'].lower() == 'gate_mpnn':
        model = GATE_MPNN(params)

    if params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    elif params['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])

    if params['loss_fn'].lower() == 'mse':
        loss_fn = nn.MSELoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        patience=10,
        factor=0.9,
        verbose=True)
    
    n_train = len(train_loader)
    n_val = len(val_loader)
    best_mae_error = 1e8

    # optionally resume from a checkpoint
    if params['resume']:
        if os.path.isfile(params['resume']):
            checkpoint = torch.load(params['resume'])
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("no checkpoint found at {}".format(params['resume']))

    for i in range(params['n_epoch']):
        outputs_val = []
        t = time.time()

        loss_train = 0.
        loss_val = 0.

        model.train()
        for batch in train_loader:
            X_batch, A_batch, E_batch, N_batch, Y_batch = batch
            optimizer.zero_grad()
            Y_preds = model(X_batch, A_batch, E_batch, N_batch)
            loss = loss_fn(Y_preds.squeeze(), 
                        Y_batch)
            loss.backward()
            optimizer.step()
            loss_train += loss.data

        model.eval()
        for batch in val_loader:
            X_batch, A_batch, E_batch, N_batch, Y_batch = batch
            Y_preds = model(X_batch, A_batch, E_batch, N_batch)
            loss = loss_fn(Y_preds.squeeze(),
                        Y_batch)
            loss.backward()
            loss_val += loss.data

            for Y_pred, Y in zip(Y_preds, Y_batch):
                outputs_val.append([Y_pred.detach().numpy(),
                                    Y.detach().numpy()])

        mae_error = cal_loss(np.array(outputs_val)*params['norm_value'], kind='mae')
        
        scheduler.step(loss_val)
        #model.eval()
        print('Epoch: {:04d}'.format(i+1),
            'loss_train: {:.4f}'.format(loss_train/n_train),
            'loss_val: {:.4f}'.format(loss_val/n_val),
            'MAE_val: {:.4f}'.format(mae_error),
            'time: {:.4f}s'.format(time.time() - t))

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        if is_best:
            save_best_model({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'params': params,
            })

    # test best model
    best_model = torch.load(params['best_model'])
    model.load_state_dict(best_model['state_dict'])
    model.eval()
    outputs_test = []
    for batch in test_loader:
        X_batch, A_batch, E_batch, N_batch, Y_batch = batch
        Y_preds = model(X_batch, A_batch, E_batch, N_batch)
        loss = loss_fn(Y_preds.squeeze(),
                    Y_batch)
        loss.backward()
        loss_val += loss.data

        for Y_pred, Y in zip(Y_preds, Y_batch):
            outputs_test.append([Y_pred.detach().numpy(),
                                Y.detach().numpy()])   

    return(np.array(outputs_test))

def cal_loss(x, kind):
    if kind.lower() == 'rmse':
        return(metrics.mean_squared_error(x[:,0], x[:,1], squared=False))

    elif kind.lower() == 'mae':
        return(metrics.mean_absolute_error(x[:,0], x[:,1]))

def save_best_model(state, filename=params['best_model']):
    torch.save(state, filename)


# Train model
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

t_total = time.time()

dataset, train_loader, val_loader, test_loader = load_data(params)
params['norm_value'] = dataset.norm_value 
params['n_node'], params['n_node_fea'] = dataset.Xs[0].shape
params['n_edge_fea'] = len(list(dataset.Es[0].values())[0])

outputs_test = train(train_loader, val_loader, test_loader, params)
outputs_test *= params['norm_value']
losses_val = cal_loss(outputs_test, kind='mae')


print('Test error: {} +- {}'.format(losses_val, 0))

x = np.linspace(-0.2*params['norm_value'], params['norm_value'], 100)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(outputs_test[:,0], outputs_test[:,1], s=10, alpha=.5)
ax.plot(x, x)
ax.set_xlabel('Prediction (kcal/mol)')
ax.set_ylabel('Experiment (kcal/mol)')
plt.show()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))