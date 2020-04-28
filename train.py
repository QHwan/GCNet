from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import GCN
from data import load_data


# Training settings
parser = argparse.ArgumentParser()
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default='GCN',
                    help='Network Model: NN, GCN')
parser.add_argument('--log', type=str, default=None,
                    help='Log file containing training process')
parser.add_argument('--o', type=str, default=None,
                    help='Save model')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')

args = parser.parse_args()


np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))


# Load data
adj_train, adj_val, adj_test, features_train, features_val, features_test, labels_train, labels_val, labels_test = load_data()

# Model and optimizer
n_feat = features_train[0].shape[-1]
model = GCN(n_feat=n_feat,
            n_hid=args.hidden,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

n_train = len(adj_train)
n_val = len(adj_val)
n_test = len(adj_test)

loss_fn = torch.nn.MSELoss(size_average=False)

def train(epoch):
    t = time.time()

    running_loss_train = 0

    output = []

    for i in range(n_train):
        model.train()
        optimizer.zero_grad()
        output_train = model(features_train[i], adj_train[i])
        loss_train = loss_fn(output_train, labels_train[i])
        model.zero_grad()
        loss_train.backward()
        optimizer.step()
        running_loss_train += loss_train.data

        output.append(output_train)

    running_loss_val = 0

    '''
    for i in range(n_val):
        model.eval()
        output_val = model(features_val[i], adj_val[i])
        loss_val = loss_fn(output_val, labels_val[i])
        running_loss_val += loss_val.data
    '''

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(running_loss_train/n_train),
          'loss_val: {:.4f}'.format(running_loss_val/n_val),
          'time: {:.4f}s'.format(time.time() - t))

    return [epoch+1, running_loss_train/n_train, running_loss_val/n_val, output]




# Train model
log = []
t_total = time.time()
for epoch in range(args.epochs):
    log_epoch = train(epoch)

output = log_epoch[-1]
out = []
for i in range(len(output)):
    out.append([labels_train[i].detach().numpy(), output[i].detach().numpy()])
out = np.array(out)

plt.scatter(out[:,0], out[:,1], s=5, alpha=.5)
x = np.linspace(-1, 1, 100)
plt.plot(x, x)
plt.show()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Save File
if args.log is not None:
    np.savetxt(args.log, log)

if args.o is not None:
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, args.o)
