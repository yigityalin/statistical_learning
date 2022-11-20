from abc import ABC, abstractmethod
from copy import deepcopy

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

    def __repr__(self) -> str:
        """
        The string representation of the activation
        :return: the string representation
        """
        return str(self.__class__).strip('>').split(' ')[-1].strip("'")

    def __str__(self) -> str:
        """
        The string representation of the activation
        :return: the string representation
        """
        return repr(self)


class Model(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Predicts the target vector given the feature matrix
        :param X: the feature matrix
        :return: predictions
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Trains the model given a dataset
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target vector given the feature matrix
        :param X: the feature matrix
        :return: predictions
        """
        pass

    def copy(self) -> 'Model':
        """
        Creates of a deepcopy of the model.
        Wraps Python's copy.deepcopy function
        :return: a copy of the model
        """
        return deepcopy(self)
