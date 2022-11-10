from typing import Callable

import numpy as np


def check_dims(X=None, y=None):
    """
    Checks the dimensions of the feature matrix and the target vector.
    :param X: the feature matrix
    :param y: the target vector
    :return: the feature matrix and the target vector
    """
    if X is not None and X.ndim != 2:
        raise ValueError('The number of dimensions of the feature matrix has to be 2.')
    if y is not None:
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        elif y.ndim != 2:
            raise ValueError('The shape of the target matrix has to be (n, 1) or (n,),'
                             ' where n is the number of the training samples')
    return X, y

def initialize_parameters(b_shape: tuple, W_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random weights from uniform distribution between 0 and 1
    :param b_shape: shape of the bias vector
    :param W_shape: shape of the weight matrix
    :return: the bias vector and the weight matrix
    """
    rng = np.random.default_rng()
    b = rng.uniform(-1, 1, b_shape)
    W = rng.uniform(-1, 1, W_shape)
    return b, W


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize the features
    :param X: the feature matrix
    :return: the normalized feature matrix
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
