#!/usr/bin/env python
# coding: utf-8

import torch

def sigmoid(yhat):
    return 1. / (1. + torch.exp(-yhat))

def nll(y, yhat, dcp=False):
    if dcp:
        return -torch.sum(y * yhat - torch.log(1 + torch.exp(yhat)))
    else:
        log_yhat = torch.log(1 + torch.exp(-yhat))
        log_one_yhat = torch.log(1 + torch.exp(yhat))
        ones_loss = y * log_yhat
        zeros_loss = (1 - y) * log_one_yhat
        return torch.sum(ones_loss + zeros_loss)