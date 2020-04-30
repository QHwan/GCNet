import numpy as np
import scipy.sparse as sp
import torch


def load_data():
    print('Loading ice dataset...')

    filename = 'dataset/freesolv.pt'
    feature, adj, label = torch.load(filename)
    norm = abs(min(label))
    for i, l in enumerate(label):
        label[i] /= norm
    n_conf = len(feature)

    adj_train_list = []
    feature_train_list = []
    label_train_list = []

    adj_val_list = []
    feature_val_list = []
    label_val_list = []

    adj_test_list = []
    feature_test_list = []
    label_test_list = []

    train_size = int(n_conf*.8)
    val_size = int(n_conf*.1)
    test_size = n_conf - train_size - val_size

    if train_size<1 or val_size<1 or test_size<1:
        print("Dataset is too small to split train-validation-test set!")
        exit(1)

    idx = np.linspace(0, n_conf-1, n_conf)
    np.random.shuffle(idx)
    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size+val_size]
    idx_test = idx[train_size+val_size:]

    for i, (f, a, l) in enumerate(zip(feature, adj, label)):

        if i in idx_train:
            adj_train_list.append(a)
            feature_train_list.append(f)
            label_train_list.append(l)

        elif i in idx_val:
            adj_val_list.append(a)
            feature_val_list.append(f)
            label_val_list.append(l)

        elif i in idx_test:
            adj_test_list.append(a)
            feature_test_list.append(f)
            label_test_list.append(l)

    return(adj_train_list, adj_val_list, adj_test_list, 
           feature_train_list, feature_val_list, feature_test_list,
           label_train_list, label_val_list, label_test_list)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def output_to_label(output):
    preds = output.max(1)[1]
    preds = preds.detach().numpy()
    return preds

def prepare_svm_data(features_data, labels_data):
    x = []
    y = []
    for feature in features_data:
        feature = feature.numpy()
        for f in feature:
            x.append(f)
    for label in labels_data:
        label = label.numpy()
        for l in label:
            y.append(l)
    return x, y

