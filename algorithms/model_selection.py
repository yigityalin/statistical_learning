from collections import namedtuple
from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd

from .base import Model
from . import metrics as m


DEFAULT_METRICS = {
    'MSE': m.mse,
    'MAE': m.mae,
    'MAPE': m.mape,
    'R2': m.r2,
}


class KFoldGridSearch:
    def __init__(self, k, model, param_grid):
        self.k = k
        self.model = model
        self.param_grid = list(param_grid)

    def cv(self,
           X: np.ndarray,
           y: np.ndarray,
           shuffle: bool = True,
           **fit_params) -> list:
        """
        Apply k-fold cross validation with given dataset
        :param X: the feature matrix
        :param y: the target vector
        :param shuffle: whether to shuffle the data before cross validation
        :param fit_params: the keyword arguments to pass to the model's fit method
        :return: the KFold instance itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        indices = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
        fold_size = len(y) // self.k
        scores = []
        for params in self.param_grid:
            model = self.model(*params)
            fold_scores = []
            for fold in range(self.k):
                valid_indices = indices[fold * fold_size: (fold + 1) * fold_size]
                train_indices = indices[~np.isin(indices, valid_indices)]

                X_train, y_train = X[train_indices], y[train_indices]
                X_valid, y_valid = X[valid_indices], y[valid_indices]
                history = model.fit(X_train, y_train, X_valid, y_valid, cold_start=True, **fit_params)
                fold_scores.append(history)
            scores.append(history)
        return scores
