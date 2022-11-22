#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sklearn.datasets

from utils.preprocess import preprocess_data
from utils.loss import nll
from utils.helpers import generate_sign_patterns, get_out_string
from utils.cvxpy_model import cvxpy_solver
from utils.pytorch_model import sgd_solver
from utils.visualization import get_times_epoch_xaxis, plot_metrics_over_time


def run_all_models(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                   out_dir, D, P, u_vector_list, num_neurons, num_epochs,
                   batch_size, beta_noncvx, learning_rate, i, verbose):
    # NN
    print('Running Neural Network...')
    solver_type = "sgd" # pick: "sgd" or "LBFGS"
    LBFGS_param = [10, 4] # these parameters are for the LBFGS solver
    results_noncvx = sgd_solver(X_train, y_train, X_valid, y_valid, 
                                num_epochs, num_neurons, beta_noncvx, 
                                learning_rate, batch_size, solver_type, 
                                LBFGS_param, D, verbose=verbose, eps=1e-2, 
                                last_n=10)

    # PyTorch - hinge loss
    print('Running PyTorch Hinge...')
    beta_cvx = 2 * beta_noncvx
    rho = 1e-4
    solver_type = "sgd" # pick: "sgd" or "LBFGS"
    LBFGS_param = [10, 4] # these parameters are for the LBFGS solver
    learning_rate = 1e-3
    results_pt_hinge = sgd_solver(X_train, y_train, X_valid, y_valid, 
                                  num_epochs, num_neurons, beta_cvx,
                                  learning_rate, batch_size, solver_type, 
                                  LBFGS_param, D, rho=rho, convex=True,
                                  u_vector_list=u_vector_list, verbose=verbose,
                                  eps=1e-2, last_n=10)

    # PyTorch - relaxed
    print('Running PyTorch Relaxed...')
    beta_cvx = 2 * beta_noncvx
    solver_type = "sgd" # pick: "sgd" or "LBFGS"
    LBFGS_param = [10, 4] # these parameters are for the LBFGS solver
    learning_rate = 1e-3
    results_pt_relaxed = sgd_solver(X_train, y_train, X_valid, y_valid, 
                                    num_epochs, num_neurons, beta_cvx,
                                    learning_rate, batch_size, solver_type, 
                                    LBFGS_param, D, rho=0, convex=True,
                                    u_vector_list=u_vector_list, verbose=verbose, 
                                    eps=1e-2, last_n=10)

    # CVXPY - exact
    print('Running CVXPY Exact...')
    max_iters = 2000
    solver_type = 'SCS' #'ECOS', 'OSQP', or 'SCS'
    beta_cvx = 2 * beta_noncvx
    batch_size = 1000
    results_cp_exact = cvxpy_solver(X_train, y_train, X_valid, y_valid, 
                                    max_iters, num_neurons, beta_cvx, 
                                    solver_type, D, u_vector_list, 
                                    batch_size=batch_size, verbose=True)

    # CVXPY - relaxed
    print('Running CVXPY Relaxed...')
    # max_iters = 100
    solver_type = 'SCS' #'ECOS', 'OSQP', or 'SCS'
    beta_cvx = 2 * beta_noncvx
    batch_size = 100000
    results_cp_relaxed = cvxpy_solver(X_train, y_train, X_valid, y_valid, 
                                      max_iters, num_neurons, beta_cvx, 
                                      solver_type, D, u_vector_list,
                                      batch_size=batch_size,
                                      exact=False, verbose=True)

    # write results to file
    print('Writing to file...\n')
    times_nc, epoch_times_nc, xaxis_nc = get_times_epoch_xaxis(results_noncvx, num_epochs)
    times_pth, epoch_times_pth, xaxis_pth = get_times_epoch_xaxis(results_pt_hinge, num_epochs)
    times_ptr, epoch_times_ptr, xaxis_ptr = get_times_epoch_xaxis(results_pt_relaxed, num_epochs)
    out_str = get_out_string(results_noncvx, results_pt_hinge, results_pt_relaxed, 
                             results_cp_exact, results_cp_relaxed, epoch_times_nc, 
                             epoch_times_pth, epoch_times_ptr, X_test, y_test, D)
    file = open(out_dir + str(i) + '.txt', 'w')
    file.write(out_str)
    file.close()

def run_dataset(data, num_runs):
    # load data
    if data == 'adult':
        base_dir = 'all_data/vector_data/adult/a5a_'
        output_dir = 'outputs/adult/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.libsvm' for ds in names]
        d_tuples = [sklearn.datasets.load_svmlight_file(file) for file in files]
        X_tr, X_v, X_tst = [data[0].toarray() for data in d_tuples]
        X_tst = X_tst[:, :-1]
    elif data == 'connect4':
        base_dir = 'all_data/vector_data/connect4/connect-4_'
        output_dir = 'outputs/connect4/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.libsvm' for ds in names]
        d_tuples = [sklearn.datasets.load_svmlight_file(file) for file in files]
        X_tr, X_v, X_tst = [data[0].toarray() for data in d_tuples]
    elif data == 'dna':
        base_dir = 'all_data/vector_data/dna/dna_scale_'
        output_dir = 'outputs/dna/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.libsvm' for ds in names]
        d_tuples = [sklearn.datasets.load_svmlight_file(file) for file in files]
        X_tr, X_v, X_tst = [data[0].toarray() for data in d_tuples]
    elif data == 'mushrooms':
        base_dir = 'all_data/vector_data/mushrooms/mushrooms_'
        output_dir = 'outputs/mushrooms/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.libsvm' for ds in names]
        d_tuples = [sklearn.datasets.load_svmlight_file(file) for file in files]
        X_tr, X_v, X_tst = [data[0].toarray() for data in d_tuples]
    elif data == 'nips':
        base_dir = 'all_data/vector_data/nips/'
        base_dir += 'nips-0-12_all_shuffled_bidon_target_'
        output_dir = 'outputs/nips/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.amat' for ds in names]
        d_tuples = [np.loadtxt(file) for file in files]
        X_tr, X_v, X_tst = [data[:, :-1] for data in d_tuples]
    elif data == 'ocr':
        base_dir = 'all_data/vector_data/ocr_letters/ocr_letters_'
        output_dir = 'outputs/ocr_letters/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.txt' for ds in names]
        d_tuples = [np.loadtxt(file) for file in files]
        X_tr, X_v, X_tst = [data[:, :-1] for data in d_tuples]
    elif data == 'rcv1':
        base_dir = 'all_data/vector_data/rcv1/rcv1_all_subset.binary_'
        output_dir = 'outputs/rcv1/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '_voc_150.amat' for ds in names]
        d_tuples = [np.loadtxt(file) for file in files]
        X_tr, X_v, X_tst = [data[:, :-1] for data in d_tuples]
    elif data == 'web':
        base_dir = 'all_data/vector_data/web/w6a_'
        output_dir = 'outputs/web/'
        names = ['train', 'valid', 'test']
        files = [base_dir + ds + '.libsvm' for ds in names]
        d_tuples = [sklearn.datasets.load_svmlight_file(file) for file in files]
        X_tr, X_v, X_tst = [data[0].toarray() for data in d_tuples]

    # hyperparameters
    X_train, y_train, X_valid, y_valid, X_test, y_test, D = preprocess_data(X_tr, X_v, X_tst, r=10)
    P, verbose = 50, True # SET verbose to True to see progress
    sign_pattern_list, u_vector_list = generate_sign_patterns(X_train, P, verbose)
    num_neurons = len(sign_pattern_list)
    num_epochs, batch_size = 5000, 100
    beta_noncvx = 1e-3
    learning_rate = 1e-3
    
    for i in range(num_runs):
        print('Running Iteration ' + str(i) + '...')
        run_all_models(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                       output_dir, D, P, u_vector_list, num_neurons, num_epochs,
                       batch_size, beta_noncvx, learning_rate, i, verbose)