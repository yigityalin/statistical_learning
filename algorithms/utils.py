from typing import Callable

import numpy as np


def initialize_parameters(b_shape: tuple, W_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random weights from uniform distribution between 0 and 1
    :param b_shape: shape of the bias vector
    :param W_shape: shape of the weight matrix
    :return: the bias vector and the weight matrix
    """
    b = np.random.rand(*b_shape)
    W = np.random.rand(*W_shape)
    return b, W


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize the features
    :param X: the feature matrix
    :return: the normalized feature matrix
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
