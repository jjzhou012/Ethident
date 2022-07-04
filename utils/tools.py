#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: tools.py
@time: 2022/1/16 20:23
@desc:
'''
import torch
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold


def setup_seed(seed):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def data_split(X, Y, seeds, K):
    # get split idx
    train_splits = []
    test_splits = []
    val_splits = []

    for seed in seeds:
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        for train_val_idx, test_idx in kf.split(X=X, y=Y):
            kf_val = StratifiedKFold(n_splits=K-1, shuffle=True, random_state=seed)
            x = X[train_val_idx]
            y = Y[train_val_idx]
            for train_idx, val_idx in kf_val.split(X=x, y=y):
                test_splits.append(X[test_idx].tolist())
                train_splits.append(x[train_idx].tolist())
                val_splits.append(x[val_idx].tolist())
    for i, train_idx in enumerate(train_splits):
        assert set(train_idx + test_splits[i] + val_splits[i]) == set(X.tolist())

    return train_splits, val_splits, test_splits





class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_results = None

    def __call__(self, val_loss, results):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_results = results
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # save best result
            self.best_results = results

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"     INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('     INFO: Early stopping')
                self.early_stop = True
