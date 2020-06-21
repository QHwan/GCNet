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
from torch.autograd import Variable


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
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--loss_fn', type=str, default='mse')
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--train_ratio', type=float, default=.6)
parser.add_argument('--val_ratio', type=float, default=.2)
parser.add_argument('--n_batch', type=int, default=2,
                    help='number of mini-batch')

args = parser.parse_args()
params = vars(args)

def save_checkpoint(state, is_best, filename=params['best_model']):
    if is_best:
        torch.save(state, filename)

def train(model, optimizer, criterion, train_loader, mode):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    n_data = params['n_'+mode]

    output = []
    loss_train = 0.
    for i, batch in enumerate(train_loader):
        X, A, N, Y = batch

        X = Variable(X.to(device))
        A = Variable(A.to(device))
        Y = Variable(Y.to(device))

        if mode == 'train':
            optimizer.zero_grad()

        Y_pred = model(X, A, N)

        loss = criterion(Y_pred, Y)

        if mode == 'train':
            loss.backward()
            optimizer.step()
        
        loss_train += loss.cpu().data.numpy()

        if mode == 'test':
            Y_pred = Y_pred.squeeze().detach().numpy()
            Y = Y.squeeze().cpu().detach().numpy()
            for j in range(len(Y)):
                output.append([Y[j], Y_pred[j]]) 

    return(loss_train/n_data, np.array(output))


# Train model
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

t_total = time.time()

dataset, train_loader, val_loader, test_loader = load_data(params)
params['n_node_fea'] = dataset.n_node_fea
params['n_edge_fea'] = dataset.n_edge_fea
params['n_out_fea'] = dataset.n_out_fea
#params['y_norm'] = dataset.Y_norm

n_train = len(train_loader)
n_val = len(val_loader)

model = GCN(params).to(device)

optimizer = optim.Adam(model.parameters(), lr=params['lr'])
criterion = nn.MSELoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    'min',
    patience=10,
    factor=0.9,
    verbose=True)

if params['resume']:
    if os.path.isfile(params['resume']):
        print("=> loading pretrained model '{}'".format(params['resume']))
        checkpoint = torch.load(params['resume'])
        args.start_epoch = checkpoint['epoch']
        best_error = checkpoint['best_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(params['resume'], checkpoint['epoch']))
    else:
        print("=> no pretrained file at '{}'".format(params['resume']))

best_error = 1e8
for i in range(params['n_epoch']):
    t = time.time()

    loss_train, _  = train(model,
                        optimizer,
                        criterion,
                        train_loader,
                        mode='train'
                        )
    loss_val, _ = train(model,
                        optimizer,
                        criterion,
                        val_loader,
                        mode='val'
                        )

    error = loss_val
    is_best = error < best_error
    best_error = min(error, best_error)
    save_checkpoint({
        'epoch': i + 1,
        'state_dict': model.state_dict(),
        'best_error': best_error,
        'optimizer': optimizer.state_dict(),
        'params': params
    }, is_best)

    print('Epoch: {} \tTime: {:.6f} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(
        i, time.time()-t, loss_train, loss_val))

if params['test']:
    loss_test, output_test = train(model,
        optimizer,
        criterion,
        test_loader,
        mode='test'
        )
    print("Test Loss: {:.6f}".format(loss_test))
    print(metrics.mean_squared_error(output_test[:,0], output_test[:,1]))
    x = np.linspace(output_test.min(), output_test.max(), 100)
    plt.plot(x, x)
    plt.scatter(output_test[:,0], output_test[:,1], s=5, alpha=.8)
    plt.show()