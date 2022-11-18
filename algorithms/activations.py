import numpy as np

from .base import Activation


class Linear(Activation):
    """
    The linear activation function.
    Implements f(x) = x.
    """
    def __call__(self, V):
        return V

    def backward(self, V, dZ):
        return dZ


class ReLU(Activation):
    """
    The Rectified Linear Unit (ReLU) activation function.
    Implements f(x) = x for x > 0 and 0 otherwise.
    """
    def __call__(self, V):
        return np.maximum(0, V)

    def backward(self, V, dZ):
        return np.where(V > 0, dZ, 0)