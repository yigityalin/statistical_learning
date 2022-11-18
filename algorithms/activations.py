from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """
    Abstract Base Class for activation functions.
    """
    @abstractmethod
    def __call__(self, V):
        """
        Implements the activation function for the forward pass.
        :param V: The induced local field of the network layer
        :return: The output Z of the activation function
        """
        pass

    @abstractmethod
    def backward(self, V, dZ):
        """
        Implements the derivative of the activation function for the backward pass.
        :param V: The induced local field of the network layer after the forward pass
        :param dZ: The derivative of the layer output
        :return: the gradient of the cost with respect to V
        """
        pass


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