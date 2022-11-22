#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import torch
from torch.autograd import Variable
from utils.helpers import get_dataloader, loss_func, validation, print_metrics, print_zero_one_split

# functions for solving the convex problem

class custom_cvx_layer(torch.nn.Module):
    def __init__(self, d, num_neurons, u_vector_list):
        """
        In the constructor we instantiate two nn.Linear modules
        and assign them as member variables.
        """
        super(custom_cvx_layer, self).__init__()
        
        self.v = torch.nn.Parameter(data=torch.zeros(d, num_neurons), 
                                    requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.zeros(d, num_neurons), 
                                    requires_grad=True)
        u_vectors_as_np = np.asarray(u_vector_list).reshape((num_neurons, d)).T
        self.h = torch.nn.Parameter(data=torch.Tensor(u_vectors_as_np), 
                                    requires_grad=False)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data 
        and we must return a Tensor of output data. We can use 
        Modules defined in the constructor as well as arbitrary 
        operators on Tensors.
        """
        sign_patterns = (torch.matmul(x, self.h) >= 0)
        Xv_w = torch.matmul(x, self.v - self.w)
        DXv_w = torch.mul(sign_patterns, Xv_w)
        y_pred = torch.sum(DXv_w, dim=1, keepdim=True)
        return y_pred

def sgd_solver(A_train, y_train, A_test, y_test, num_epochs, 
               num_neurons, beta, learning_rate, batch_size, 
               solver_type, LBFGS_param, D, u_vector_list=[],
               rho=0, eps=1e-2, last_n=5, convex=False, 
               verbose=False):
    device = torch.device('cpu')

    # get dimensions of train and test data sets
    D_in, H, D_out = A_train.shape[1], num_neurons, y_train.shape[1]
    N_train = A_train.shape[0]
    N_test = A_test.shape[0]
    
    # create the model
    if convex:
        model = custom_cvx_layer(D_in, H, u_vector_list).to(device)
    else:
        model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H, D_out, bias=False),
                ).to(device)
    
    # create the optimizer
    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=0.9)
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      history_size=LBFGS_param[0], 
                                      max_iter=LBFGS_param[1])
    
    # arrays for saving the loss, accuracy, nll, and time
    losses_batch = np.zeros((int(num_epochs * np.ceil(N_train / batch_size))))
    accs_batch = np.zeros(losses_batch.shape)
    
    losses_train = np.zeros((num_epochs+1))
    accs_train = np.zeros((num_epochs+1))
    nll_train = np.zeros((num_epochs+1))
       
    losses_test = np.zeros((num_epochs+1))
    accs_test = np.zeros((num_epochs+1))
    nll_test = np.zeros((num_epochs+1))

    times = np.zeros((losses_batch.shape[0]+1))
    times[0] = time.time()
    
    fp_times = []
    loss_times = []
    bp_times = []
    update_times = []
    
    # dataset loaders (minibatch)
    ds = get_dataloader(A_train, y_train, batch_size)
    ds_test = get_dataloader(A_test, y_test, A_test.shape[0])

    # loss on the entire train & test sets
    val_tr = validation(model, ds, beta, D, N_train, rho, convex)
    losses_train[0], accs_train[0], nll_train[0] = val_tr
    val_tst = validation(model, ds_test, beta, D, N_test, rho, convex)
    losses_test[0], accs_test[0], nll_test[0] = val_tst

    # run the model
    iter_no = 0
    epoch_no = 0
    for i in range(num_epochs):
        for ix, (_x, _y) in enumerate(ds):
            #=========make input differentiable=======================
            _x = Variable(_x).float()
            _y = Variable(_y).float()

            
            ####### this function is for LBFGS ######
            def closure():
                optimizer.zero_grad()
                yhat = model(_x).float()
                loss = loss_func(yhat, _y, model, beta, _x, rho, convex)
                loss.backward()
                return loss
            #########################################
            
            
            #========forward pass=====================================
            start_fp = time.time()
            yhat = model(_x).float()
            end_fp = time.time()
            loss = loss_func(yhat, _y, model, beta, _x, rho, convex)
            end_loss = time.time()
            loss /= batch_size
            correct = torch.eq(yhat>0, _y).float().sum()

            #=======backward pass=====================================
            if solver_type == "sgd":
                # zero the gradients on each pass before the update
                optimizer.zero_grad() 
                start_bkwd = time.time()
                loss.backward() # backprop the loss through the model
                end_bkwd = time.time()
                optimizer.step() # update the gradients w.r.t the loss
                end_update = time.time()
            elif solver_type == "LBFGS":
                optimizer.step(closure)
            
            # update arrays
            losses_batch[iter_no] = loss.item() / batch_size
            accs_batch[iter_no] = correct / batch_size
            
            fp_times += [end_fp - start_fp]
            loss_times += [end_loss - end_fp]
            bp_times += [end_bkwd - start_bkwd]
            update_times += [end_update - end_bkwd]
        
            iter_no += 1
            times[iter_no] = time.time()
        
        # get train/test loss and accuracy
        val_tst = validation(model, ds_test, beta, D, N_test, rho, convex)
        losses_test[i+1], accs_test[i+1], nll_test[i+1] = val_tst
        val_tr = validation(model, ds, beta, D, N_train, rho, convex)
        losses_train[i+1], accs_train[i+1], nll_train[i+1] = val_tr
        
        epoch_no += 1
        
        if i % 1 == 0:
            train_str = print_metrics(nll_train[i+1], losses_train[i+1], 
                                      accs_train[i+1], 'TRAIN: ')
            test_str = print_metrics(nll_test[i+1], losses_test[i+1], 
                                     accs_test[i+1], '\n\t\tTEST: ')
            print_str = 'Epoch [{}/{}], ' + train_str + test_str
            print(print_str.format(i, num_epochs))
            
        if i >= last_n:
            arr1 = np.array(nll_test[i+2-last_n : i+2])
            arr2 = np.array(nll_test[i+1-last_n : i+1])
            if np.mean(arr2 - arr1) < eps:
                break

    print_zero_one_split(A_train, y_train, A_test, y_test,
                         model, beta, D, rho, convex)

    arr_train = [losses_train[:epoch_no+1], nll_train[:epoch_no+1], accs_train[:epoch_no+1]]
    arr_test = [losses_test[:epoch_no+1], nll_test[:epoch_no+1], accs_test[:epoch_no+1]]
    all_times = [fp_times, loss_times, bp_times, update_times]
    return arr_train + arr_test + [epoch_no, all_times, times[:epoch_no+1], model]
