#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import time
import scipy
from scipy.sparse.linalg import LinearOperator
import torch
import sklearn.linear_model
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

##################### TIME VISUALIZATION ##########################
def plot_breakdown_by_batch(all_times1, all_times2, start=0, end=2000):
    fpt1, losst1, bpt1, updatet1 = all_times1
    fpt2, losst2, bpt2, updatet2 = all_times2
    fpt = np.array(fpt1) - np.array(fpt2)
    losst = np.array(losst1) - np.array(losst2)
    bpt = np.array(bpt1) - np.array(bpt2)
    updatet = np.array(updatet1) - np.array(updatet2)
    plt.rcParams.update({'font.size': 16})
    plt.plot(1000*fpt[start:end], label='forward')
    plt.plot(1000*losst[start:end], label='loss')
    plt.plot(1000*bpt[start:end], label='backward')
    plt.plot(1000*updatet[start:end], label='update')
    plt.xlabel('Batch')
    plt.ylabel('Time (ms)')
    plt.legend(loc='lower left', ncol=2)

def get_step_errors(fpt, losst, bpt, updatet, stddev):
    mean = 1000 * np.array([np.mean(fpt), np.mean(losst), np.mean(bpt), np.mean(updatet)])
    if stddev:
        err = [np.std(fpt), np.std(losst), np.std(bpt), np.std(updatet)]
    else:
        min_err = [np.min(fpt), np.min(losst), np.min(bpt), np.min(updatet)]
        max_err = [np.max(fpt), np.max(losst), np.max(bpt), np.max(updatet)]
        err = 1000 * np.concatenate([np.array(min_err)[None, :], np.array(max_err)[None, :]], axis=0)
        err = np.abs(err - mean)
    
    return mean, err
    
def plot_breakdown_by_step(all_times1, all_times2, label1, label2, stddev=False):
    fpt1, losst1, bpt1, updatet1 = all_times1
    fpt2, losst2, bpt2, updatet2 = all_times2
    err_x = ['fwd pass', 'loss', 'bkwd pass', 'update']
    mean1, err1 = get_step_errors(fpt1, losst1, bpt1, updatet1, stddev)
    mean2, err2 = get_step_errors(fpt2, losst2, bpt2, updatet2, stddev)
    
    capthick=5
    capsize=10
    markersize=capsize
    elinewidth=capthick
    plt.errorbar(err_x, mean1, err1, linestyle='None', marker='o', 
                 label=label1, markersize=markersize, capsize=capsize, 
                 elinewidth=elinewidth, capthick=capthick)
    plt.errorbar(err_x, mean2, err2, linestyle='None', marker='o', 
                 label=label2, markersize=markersize, capsize=capsize, 
                 elinewidth=elinewidth, capthick=capthick)
    plt.xlabel('Model training step')
    plt.ylabel('Time (ms)')
    plt.grid()
    plt.legend(fontsize=14, ncol=2, markerscale=0.25)

def get_times_epoch_xaxis(results, num_epochs, cvxpy=False, time_no=-2, epoch_no=-4):
    num_epochs = int(results[epoch_no])
    iters = range(num_epochs + 1)
    times = results[time_no]

    # epoch times
    if cvxpy:
        epoch_times = [times] * (num_epochs + 1)
    else:
        N = len(times) - 1
        it = N // num_epochs
        epoch_times = times[0 : N+1 : it] - times[0]

    xaxis = [epoch_times, iters]
    return times, epoch_times, xaxis

def plot_batch_times(times_nc, times_pth, times_ptr):
    plt.plot(times_nc[1:] - times_nc[:-1], 'o', label='nonconvex')
    plt.plot(times_pth[1:] - times_pth[:-1], 'o', label='PT hinge')
    plt.plot(times_ptr[1:] - times_ptr[:-1], 'o', label='PT relaxed')
    plt.legend()

def plot_epoch_times(epoch_times_nc, epoch_times_pth, epoch_times_ptr):
    plt.plot(epoch_times_nc, 'o', label='nonconvex')
    plt.plot(epoch_times_pth, 'o', label='PT hinge')
    plt.plot(epoch_times_ptr, 'o', label='PT relaxed')
    plt.legend()


# plot negative log likelihood by time (sec) for validation
# x_ax_no --> 0: time, 1: iteration
# plot_no --> 0: model loss, 1: nll, 2: accuracy
# data_no --> 0: training, 1: validation
def plot_metrics_over_time(xaxis_pth=[], xaxis_ptr=[], xaxis_nc=[], 
                           xaxis_cpe=[], xaxis_cpr=[], results_pt_hinge=[], 
                           results_pt_relaxed=[], results_noncvx=[], 
                           results_cp_exact=[], results_cp_relaxed=[], 
                           x_ax_no=1, plot_no=1, data_no=1, num_epochs=-2):
    plt.rcParams.update({'font.size': 16})

    data_offset = 3
    offset = data_no * data_offset

    # label x axis, y axis, title
    xlabel_list = ['Time (sec)', 'Iteration']
    xlabel = xlabel_list[x_ax_no]
    plt.xlabel(xlabel)

    ylabel_list = ["Cost", "Negative Log Likelihood", "Accuracy"]
    ylabel = ylabel_list[plot_no]
    plt.ylabel(ylabel)

    data_str = ['Training', 'Validation']
    # type_str = ylabel + ' vs ' + xlabel + ' (' + data_str[data_no] + '): '
    type_str = data_str[data_no]

    title_str = [' over Time', ' per ']
    str2 = title_str[x_ax_no] + xlabel
    # plt.title(type_str + ' ' + ylabel + str2)
    # n, d = X_train.shape
    # plt.title(type_str + ': n={}, d={}, P={}'.format(n, d, num_neurons))

    # plot results on the data set
    if results_pt_hinge != []:
        plt.plot(xaxis_pth[x_ax_no], results_pt_hinge[plot_no + offset], 
                 label="pytorch hinge")
    if results_pt_relaxed != []:
        plt.plot(xaxis_ptr[x_ax_no], results_pt_relaxed[plot_no + offset],
                 label="pytorch relaxed")
    if results_noncvx != []:
        plt.plot(xaxis_nc[x_ax_no], results_noncvx[plot_no + offset], 
                 label="nonconvex")
    if results_cp_exact != []:
        yaxis = [results_cp_exact[plot_no + offset]] * (num_epochs + 1)
        plt.plot(xaxis_cpe[x_ax_no], yaxis,
                 label="cvxpy exact")
    if results_cp_relaxed != []:
        yaxis = [results_cp_relaxed[plot_no + offset]] * (num_epochs + 1)
        plt.plot(xaxis_cpr[x_ax_no], yaxis,
                 label="cvxpy relaxed")
        
    plt.legend()
    #plt.savefig("minibatch_solver_plots/plot1.pdf", bbox_inches="tight")

##################### IMAGE VISUALIZATION ##########################
def plot_img(mnist_data, data, model, i):
    dim = 28 * 28
    start_index = i * (dim - r)
    end_index = start_index + dim - r
    mnist_image = data[start_index : end_index]
    _x = Variable(torch.Tensor(mnist_image)).float()
    with torch.no_grad():
        yhat = model(_x).float()
    r_zeros = np.zeros((r, 1))
    pred_img = np.concatenate([r_zeros, yhat.data.numpy() > 0])
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(pred_img.reshape(28, 28))
    ax[0].set_title('predicted image')
    ax[1].imshow(mnist_data[i])
    ax[1].set_title('real image')
    
def plot_nbyn(data, n):
    k = int(np.sqrt(n))
    ones_dim = 1
    imgs = []
    for j in range(k):
        row = []
        for i in range(k):
            row += [data[j*k + i]]
            row += [np.ones((28, ones_dim))]
        imgs += [np.concatenate(row[:-1], axis=1)]
        imgs += [np.ones((ones_dim, (28 + ones_dim)*k - ones_dim))]
    return np.concatenate(imgs[:-1], axis=0)

def plot_all_models(mnist_data, data, start, n, models, names, fontsize, width, scale, r=10):
    dim = 28 * 28
    start_index = start * (dim - r)
    end_index = start_index + n * (dim - r)
    mnist_image = data[start_index : end_index]
    _x = Variable(torch.Tensor(mnist_image)).float()

    with torch.no_grad():
        yhats = [model.forward(_x).float() for model in models]

    reshaped_bools = [yhat.data.numpy().reshape((n, -1)) > 0 for yhat in yhats]
    pred_imgs = [np.concatenate([np.zeros((n, r)), rb], axis=1).reshape((n, 28, 28)) for rb in reshaped_bools]
    final_imgs = [plot_nbyn(pred_img, n) for pred_img in pred_imgs]

    f, ax = plt.subplots(2, 3, figsize=(width, 3*(width/4)), dpi=width*10)
    size = ax[0, 0].get_window_extent().transformed(f.dpi_scale_trans.inverted()).height / scale
    
    ax[0, 0].imshow(plot_nbyn(mnist_data[start:], n))
    ax[0, 0].set_title('Ground Truth', fontsize=fontsize)

    ax[0, 1].imshow(final_imgs[0])
    ax[0, 1].set_title(names[0], fontsize=fontsize)

    ax[0, 2].imshow(final_imgs[1])
    ax[0, 2].set_title(names[1], fontsize=fontsize)

    ax[1, 0].imshow(final_imgs[2])
    ax[1, 0].set_title(names[2], fontsize=fontsize)#, y=-size)
    
    ax[1, 1].imshow(final_imgs[3])
    ax[1, 1].set_title(names[3], fontsize=fontsize)#, y=-size)
    
    ax[1, 2].imshow(final_imgs[4])
    ax[1, 2].set_title(names[4], fontsize=fontsize)#, y=-size)
    
    plt.subplots_adjust(wspace=0.1, hspace=0)
    
    for row in ax:
        for col in row:
            col.xaxis.set_visible(False)
            col.yaxis.set_visible(False)


