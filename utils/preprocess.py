#!/usr/bin/env python
# coding: utf-8

import numpy as np

'''
Shifts the given data by creating D vectors of size (r) 
from an input vector of size (D + r)
params
    data: array to shift, shape (N, D)
    r: amount to shift data by

returns
    shifted array, shape (N*D, r)
'''
def shift_data(data, D, r):
    N = data.shape[0]
    data_proc = np.zeros((N, D, r))
    for i in range(D):
        data_proc[:, i] = data[:, i : i+r]
    return data_proc.reshape((N * D, r))

def get_shifted_data_labels(X, D, r):
    X_shifted = shift_data(X, D, r)
    N = X_shifted.shape[0]
    y_shifted = (X[:, r:]).reshape(N, 1)
    return X_shifted, y_shifted

def balance_data(shifted, labels):
    ones = (labels == 1)[:, 0]
    zeros = (labels == 0)[:, 0]
    num_ones = np.sum(labels == 1)
    num_zeros = np.sum(labels == 0)
    zero_indices = np.random.choice(num_zeros, size=(num_ones), replace=False)
    labels_zeros = labels[zeros][zero_indices]
    shifted_zeros = shifted[zeros][zero_indices]
    labels_ones = labels[ones]
    shifted_ones = shifted[ones]
    X = np.concatenate([shifted_ones, shifted_zeros])
    y = np.concatenate([labels_ones, labels_zeros])
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return X, y

def preprocess_data(X_tr, X_v, X_tst=[], r=10):
    D = X_tr.shape[1] - r
    shifted, labels = get_shifted_data_labels(X_tr, D, r)
    X_train, y_train = balance_data(shifted, labels)
    X_valid, y_valid = get_shifted_data_labels(X_v, D, r)
    if X_tst == []:
        return X_train, y_train, X_valid, y_valid, D
    X_test, y_test = get_shifted_data_labels(X_tst, D, r)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, D 
