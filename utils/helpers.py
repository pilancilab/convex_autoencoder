#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from utils.loss import nll

'''
Generate diagonal matrices D_i for convex models.
'''
def check_if_already_exists(element_list, element):
    # check if element exists in element_list
    # where element is a numpy array
    for i in range(len(element_list)):
        if np.array_equal(element_list[i], element):
            return True
    return False

def generate_sign_patterns(A, P, verbose=False): 
    # generate sign patterns
    n, d = A.shape
    unique_sign_pattern_list = []  # sign patterns
    u_vector_list = [] # random vectors to generate the sign patterns

    for i in range(P): 
        # obtain a sign pattern
        u = np.random.normal(0, 1, (d,1)) # sample u
        sampled_sign_pattern = (np.matmul(A, u) >= 0)[:,0]

        # check whether that sign pattern has already been used
        if not check_if_already_exists(unique_sign_pattern_list, 
                                       sampled_sign_pattern):
            unique_sign_pattern_list.append(sampled_sign_pattern)
            u_vector_list.append(u)

    if verbose:
        print("Number of unique sign patterns generated: " + 
              str(len(unique_sign_pattern_list)))
    return unique_sign_pattern_list, u_vector_list

'''
Build dataloader for pytorch models.
'''
class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(A, y, batch_size):
    data = PrepareData(X=A, y=y)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

'''
PyTorch model loss function and validation helpers
'''
def loss_func_cvxproblem(yhat, y, model, _x, beta, rho):
    # NLL loss
    loss = nll(y, yhat)
    
    # regularization
    loss = loss + beta * torch.sum(torch.norm(model.v, dim=0))
    loss = loss + beta * torch.sum(torch.norm(model.w, dim=0))
    
    if rho == 0:
        return loss
    
    # hinge loss
    sign_patterns = (torch.matmul(_x, model.h) >= 0)
    
    Xv = torch.matmul(_x, model.v)
    DXv = torch.mul(sign_patterns, Xv)
    relu_term_v = torch.max(-2*DXv + Xv, torch.Tensor([0])) # had: [0]
    loss = loss + rho * torch.sum(relu_term_v)
    
    Xw = torch.matmul(_x, model.w)
    DXw = torch.mul(sign_patterns, Xw)
    relu_term_w = torch.max(-2*DXw + Xw, torch.Tensor([0]))
    loss = loss + rho * torch.sum(relu_term_w)
    
    return loss

def loss_func_noncvxproblem(yhat, y, model, beta):
    loss = nll(y, yhat)
    for p in model.parameters():
        loss = loss + beta/2 * torch.norm(p)**2
    return loss

def loss_func(yhat, y, model, beta, x=0, rho=0, convex=False):
    if convex:
        return loss_func_cvxproblem(yhat, y, model, x, beta, rho)
    else:
        return loss_func_noncvxproblem(yhat, y, model, beta)

def validation(model, testloader, beta, D, N, rho=0, convex=False):
    test_loss = 0
    test_correct = 0
    test_nll_cost = 0

    for ix, (_x, _y) in enumerate(testloader):
        _x = Variable(_x).float()
        _y = Variable(_y).float()

        with torch.no_grad():
            yhat = model(_x).float()
            loss = loss_func(yhat, _y, model, beta, _x, rho, convex)
            test_loss += loss.item()
            test_correct += torch.eq(yhat>0, _y).float().sum()
            test_nll_cost += nll(_y, yhat)

    return test_loss * D / N, test_correct / N, test_nll_cost * D / N

def get_test_nll(results_noncvx, results_pt_hinge, results_pt_relaxed, 
                 results_cp_exact, results_cp_relaxed, X_test, y_test, D):
    dn = D / X_test.shape[0]
    yhat_test = results_noncvx[-1].forward(torch.Tensor(X_test))
    nc_nll = nll(torch.Tensor(y_test), yhat_test).data.numpy() * dn

    yhat_test = results_pt_hinge[-1].forward(torch.Tensor(X_test))
    pth_nll = nll(torch.Tensor(y_test), yhat_test).data.numpy() * dn

    yhat_test = results_pt_relaxed[-1].forward(torch.Tensor(X_test))
    ptr_nll = nll(torch.Tensor(y_test), yhat_test).data.numpy() * dn

    _, cpe_nll, _ = results_cp_exact[-1].get_metrics(X_test, y_test, D)
    _, cpr_nll, _ = results_cp_relaxed[-1].get_metrics(X_test, y_test, D)
    
    return nc_nll, pth_nll, ptr_nll, cpe_nll, cpr_nll

'''
PyTorch model print helpers
'''
def format_string(nc, cpe, pth, cpr, ptr, prefix):
    nc = np.round(nc, 3)
    cpe = np.round(cpe, 3)
    pth = np.round(pth, 3)
    cpr = np.round(cpr, 3)
    ptr = np.round(ptr, 3)
    return prefix + 'nc {}, cpe {}, pth {}, cpr {}, ptr {}'.format(nc, cpe, pth, cpr, ptr)

def get_out_string(results_noncvx, results_pt_hinge, results_pt_relaxed, 
                   results_cp_exact, results_cp_relaxed, epoch_times_nc, 
                   epoch_times_pth, epoch_times_ptr, X_test, y_test, D, 
                   last_n=10, loss_no=1, time_no=-2, iter_no=-4):
    test_nll = get_test_nll(results_noncvx, results_pt_hinge, results_pt_relaxed, 
                            results_cp_exact, results_cp_relaxed, X_test, y_test, D)
    nc_nll, pth_nll, ptr_nll, cpe_nll, cpr_nll = test_nll
    test_str = format_string(nc_nll, cpe_nll, pth_nll, cpr_nll, ptr_nll, 'test: ')
    
    train_str = format_string(results_noncvx[loss_no][-last_n], 
                              results_cp_exact[loss_no], 
                              results_pt_hinge[loss_no][-last_n], 
                              results_cp_exact[loss_no], 
                              results_pt_relaxed[loss_no][-last_n], 
                              '\ntrain: ')
    time_str = format_string(epoch_times_nc[-last_n], 
                             results_cp_exact[time_no], 
                             epoch_times_pth[-last_n], 
                             results_cp_exact[-time_no], 
                             epoch_times_ptr[-last_n], 
                             '\ntime: ')
    iter_str = format_string(results_noncvx[iter_no], 
                             results_cp_exact[time_no], 
                             results_pt_hinge[iter_no], 
                             results_cp_exact[time_no], 
                             results_pt_relaxed[iter_no], 
                             '\niter: ')
    out_str = test_str + train_str + time_str + iter_str
    return out_str

def print_metrics(nll, loss, acc, prefix):
    nll = np.round(nll, 3)
    loss = np.round(loss, 3)
    acc = np.round(acc, 3)
    metrics_str = 'nll: {}, loss: {}, acc: {}.'.format(nll, loss, acc)
    return prefix + metrics_str
    
def val_and_print(model, ds, beta, rho, convex, D, N, prefix):
    loss, acc, nll = validation(model, ds, beta, D, N, rho, convex)
    return print_metrics(nll, loss, acc, prefix)

def print_zero_one_split(A_train, y_train, A_test, y_test, model, 
                         beta, D, rho=0, convex=False):
    # get zero and one indices for train/test
    indices_train_zeros = [y_train == 0][0][:, 0]
    indices_train_ones = [y_train == 1][0][:, 0]
    indices_test_zeros = [y_test == 0][0][:, 0]
    indices_test_ones = [y_test == 1][0][:, 0]
    
    # get zero and one A, y for train/test
    A_train_zeros = A_train[indices_train_zeros]
    y_train_zeros = y_train[indices_train_zeros]
    A_train_ones = A_train[indices_train_ones]
    y_train_ones = y_train[indices_train_ones]
    A_test_zeros = A_test[indices_test_zeros]
    y_test_zeros = y_test[indices_test_zeros]
    A_test_ones = A_test[indices_test_ones]
    y_test_ones = y_test[indices_test_ones]
    
    # get zero and one counts for train/test
    N_train_zeros = A_train_zeros.shape[0]
    N_train_ones = A_train_ones.shape[0]
    N_test_zeros = A_test_zeros.shape[0]
    N_test_ones = A_test_ones.shape[0]
    
    # get dataloaders
    ds_zeros_train = get_dataloader(A_train_zeros, y_train_zeros, N_train_zeros)
    ds_ones_train = get_dataloader(A_train_ones, y_train_ones, N_train_ones)
    ds_zeros_test = get_dataloader(A_test_zeros, y_test_zeros, N_test_zeros)
    ds_ones_test = get_dataloader(A_test_ones, y_test_ones, N_test_ones)

    # print validation results
    zeros_str = val_and_print(model, ds_zeros_train, beta, rho, 
                              convex, D, N_train_zeros, 'ZEROS ')
    ones_str = val_and_print(model, ds_ones_train, beta, rho, 
                             convex, D, N_train_ones, '\n\tONES ')
    train_str = '\nTRAIN:  ' + zeros_str + ones_str
    
    zeros_str = val_and_print(model, ds_zeros_test, beta, rho, 
                              convex, D, N_test_zeros, 'ZEROS ')
    ones_str = val_and_print(model, ds_ones_test, beta, rho, 
                             convex, D, N_test_ones, '\n\tONES ')
    test_str = '\nTEST:   ' + zeros_str + ones_str
    print(train_str + test_str)
