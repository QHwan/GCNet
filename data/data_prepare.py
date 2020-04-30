import random
import numpy as np
import torch

def data_prepare(filename,
                train_ratio=.8, val_ratio=.1, test_ratio=.1):
    f = np.load(filename, allow_pickle=True)

    Xs = f['Xs']
    As = f['As']
    Ys = f['Ys']

    num_data = len(Ys)
    num_train = int(num_data*train_ratio)
    num_val = int(num_data*val_ratio)
    num_test = num_data - num_train - num_val

    idx = list(range(num_data))
    random.shuffle(idx)
    idx_train = idx[:num_train]
    idx_val = idx[num_train:num_train+num_val]
    idx_test = idx[num_train+num_val:]

    Xs_train = Xs[idx_train]
    Xs_val = Xs[idx_val]
    Xs_test = Xs[idx_test]

    As_train = As[idx_train]
    As_val = As[idx_val]
    As_test = As[idx_test]

    Ys_train = Ys[idx_train]
    Ys_val = Ys[idx_val]
    Ys_test = Ys[idx_test]

    # Y normalization
    if abs(max(Ys)) > abs(min(Ys)):
        norm_value = max(Ys)
    else:
        norm_value = min(Ys)
    Ys_train /= norm_value
    Ys_val /= norm_value
    Ys_test /= norm_value

    print(Ys_train)

    return(((Xs_train, Xs_val, Xs_test),
            (As_train, As_val, As_test),
            (Ys_train, Ys_val, Ys_test)),
           norm_value)

if __name__ == "__main__":
    data_prepare(filename='dataset/freesolv.npz',
                train_ratio=.8,
                val_ratio=.1,
                test_ratio=.1)