#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cvxpy as cp
import time
from utils.helpers import print_metrics
import torch

class cvxpy_model():
    def __init__(self, Uopt1, Uopt2, u_vectors_as_np, reg):
        """
        In the constructor we instantiate two nn.Linear modules
        and assign them as member variables.
        """        
        self.Uopt1 = Uopt1
        self.Uopt2 = Uopt2
        self.u_vectors_as_np = u_vectors_as_np
        self.reg = reg

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data 
        and we must return a Tensor of output data. We can use 
        Modules defined in the constructor as well as arbitrary 
        operators on Tensors.
        """
        tensor = False
        if torch.is_tensor(x):
            x = x.data.numpy()
            tensor = True
        dmat = (x @ self.u_vectors_as_np) >= 0
        yopt1 = np.sum(np.multiply(dmat, (x @ self.Uopt1)), axis=1)
        yopt2 = np.sum(np.multiply(dmat, (x @ self.Uopt2)), axis=1)
        if tensor:
            return torch.Tensor(yopt1 - yopt2)
        return yopt1 - yopt2
    
    def get_metrics(self, x, y, D):
        # get yhat
        yhat = self.forward(x)[:, None]
        N = yhat.shape[0]
        
        # calculate nll
        linear_term = np.multiply(y, yhat)
        logistic_term = np.log(1 + np.exp(yhat))
        nll = -np.sum(linear_term - logistic_term)
        
        # calculate nll, loss, accuracy
        loss = (nll + self.reg) * D / N
        acc = np.sum((yhat > 0) == y) / N
        nll = nll * D / N
        
        return loss, nll, acc

def cvxpy_solver(X_train, y_train, X_test, y_test, max_iters, 
                 m1, beta_cvx, solver_type, D, u_vector_list,
                 batch_size=-1, num_epochs=1, exact=True, 
                 verbose=True):
#     spl_new = [spl[:, None] for spl in sign_pattern_list]
#     dmat = np.concatenate(spl_new, axis=1)
#     if batch_size > 0:
#         X_train = X_train[:batch_size]
#         y_train = y_train[:batch_size]
#         dmat = dmat[:batch_size]
    if batch_size > 0:
        X_train = X_train[:batch_size]
        y_train = y_train[:batch_size]
    n, d = X_train.shape            
    u_vectors_as_np = np.asarray(u_vector_list).reshape((m1, d)).T
    dmat = (X_train @ u_vectors_as_np) >= 0
    Uopt1 = cp.Variable((d, m1))
    Uopt2 = cp.Variable((d, m1))

    # Calculate yopt1, yopt2
    yopt1 = cp.Parameter((n, 1))
    yopt2 = cp.Parameter((n, 1))
    yopt1 = cp.sum(cp.multiply(dmat, (X_train @ Uopt1)), axis=1)
    yopt2 = cp.sum(cp.multiply(dmat, (X_train @ Uopt2)), axis=1)

    # negative log-likelihood
    nll = cp.Parameter()
    yhat_opt = (yopt1 - yopt2)[:, None]
    # ones_term = cp.multiply(y_train, cp.log(1 + cp.exp(-yhat_opt)))
    # zeros_term = cp.multiply(1 - y_train, cp.log(1 + cp.exp(yhat_opt)))
    # bce = cp.sum(ones_term + zeros_term)
    # bce = (ones_term + zeros_term)[0]
    nll = -cp.sum(cp.multiply(y_train, yhat_opt) - cp.logistic(yhat_opt))

    # regularization
    norm1 = cp.mixed_norm(Uopt1.T, 2, 1)
    norm2 = cp.mixed_norm(Uopt2.T, 2, 1)
    reg = beta_cvx * (norm1 + norm2)
    cost = (nll + reg) * D / n

    # constraints
    constraints = []
    dmat_ones = 2 * dmat - np.ones((n, m1))
    for i in range(m1):
        constraints += [cp.multiply(dmat_ones, (X_train * Uopt1)) >= 0]
        constraints += [cp.multiply(dmat_ones, (X_train * Uopt2)) >= 0]
    
    # create and run problem
    if exact:
        prob = cp.Problem(cp.Minimize(cost), constraints)
    else:
        prob = cp.Problem(cp.Minimize(cost), [])
    print('created problem')
    start = time.time()
    prob.solve(solver=solver_type, max_iters=max_iters, verbose=verbose)
    end = time.time()
    cvx_opt = prob.value
    print("Convex program objective value (eq (8)):", cvx_opt)
    time_cvxe = end - start
    print('Time:', time_cvxe)
    
    # print results and return array
    yhat_train = (yopt1 - yopt2).value
    acc_train = np.sum((yhat_train > 0) == y_train) / n
    arr_train = [cvx_opt, nll.value * D / n, acc_train]
    
    model = cvxpy_model(Uopt1.value, Uopt2.value, u_vectors_as_np, reg.value)
    arr_test = [i for i in model.get_metrics(X_test, y_test, D)]
    train_str = print_metrics(arr_train[1], arr_train[0], arr_train[2], 'TRAIN: ')
    test_str = print_metrics(arr_test[1], arr_test[0], arr_test[2], '\nTEST: ')
    print(train_str + test_str)
    return arr_train + arr_test + [time_cvxe, model]

