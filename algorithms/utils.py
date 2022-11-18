from typing import Tuple

import numpy as np


def initialize_parameters(b_shape: Tuple, W_shape: Tuple) -> Tuple[np.ndarray, np.ndarray]:
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
