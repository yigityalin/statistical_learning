from numbers import Number
from typing import Callable, Iterable, List, Tuple, Type, Union

import numpy as np

from . import utils
from .activations import Activation
from .base import Model


class FullyConnectedLayer:
    """
    A fully connected layer with a specified number of neurons and an activation function.
    Attributes:
        in_features : int
            size of each input sample
        out_features : int
            size of each output sample
        activation : Activation
            the activation function of the layer
        b : np.ndarray
            the bias vector of the layer
        W : np.ndarray
            the weight matrix of the layer
        grad_b : np.ndarray
            the gradient vector of the bias of the layer
        grad_W : np.ndarray
            the gradient matrix of the weights of the layer
    """
    def __init__(self, in_features: int, out_features: int, activation: Type[Activation]):
        """
        Initializes the activation and the weights of the layer
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param activation: the activation function of the layer
        """
        self.in_features = in_features
        self.out_features = out_features
        self._activation = activation()
        self._b = None
        self._W = None
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes the layer parameters
        """
        self._b, self._W = utils.initialize_parameters(b_shape=(1, self.out_features),
                                                       W_shape=(self.in_features, self.out_features))

    @property
    def b(self) -> np.ndarray:
        """
        The bias vector of the layer
        :return: the bias vector
        """
        return self._b

    @property
    def W(self) -> np.ndarray:
        """
        The weight matrix of the layer
        :return: the weight matrix
        """
        return self._W

    @property
    def activation(self) -> Activation:
        """
        The activation function of the layer
        :return: the activation function
        """
        return self._activation

    def V(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the induced local field of each neuron in the layer
        :return: the induced local fields of the neurons
        """
        return self.b + X @ self.W

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass of the layer
        :param X: the feature matrix
        :return: the output of the layer
        """
        return self.activation(self.V(X))

    def __repr__(self) -> str:
        """
        The string representation of the layer
        :return: the string representation
        """
        return f'FullyConnectedLayer(in_features={self.in_features}, ' \
               f'out_features={self.out_features}, activation={self.activation})'

    def __str__(self) -> str:
        """
        The string representation of the layer
        :return: the string representation
        """
        return repr(self)

    def apply_gradients(self, alpha: Number, db: np.ndarray, dW: np.ndarray):
        """
        Applies gradient descent updates to the layer weights
        :param alpha: the learning rate
        :param db: the bias gradients
        :param dW: the weight gradients
        :return: None
        """
        self._b -= alpha * db
        self._W -= alpha * dW


class NeuralNetwork(Model):
    """
    Neural network model that is a stack of FullyConnectedLayers instances.

    Attributes:
          layers : list[FullyConnectedLayer]
                a list that contains all the network layers
    """
    def __init__(self, layers: Iterable[Tuple[int, int, Type[Activation]]]):
        """
        Initialize the layers of the network
        :param layers: the layer list that contains the number of input and output features and the activation
        """
        self._layers = [FullyConnectedLayer(n_in, n_out, activation=activation)
                        for n_in, n_out, activation in layers]
        if len(self._layers) == 0:
            raise ValueError('layers cannot be empty')

    @property
    def layers(self) -> List[FullyConnectedLayer]:
        """
        The layer list of the neural network
        :return: the list of layers of the network
        """
        return self._layers

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Wraps the model's predict method
        :param X: the feature matrix
        :return: predictions
        """
        return self.predict(X)

    def __repr__(self) -> str:
        """
        The string representation of the neural network
        :return: the string representation
        """
        return f'NeuralNetwork(n_neurons={[layer.out_features for layer in self.layers]})'

    def __str__(self) -> str:
        """
        The string representation of the neural network
        :return: the string representation
        """
        return repr(self)

    @staticmethod
    def calculate_gradients(m: int,
                            dV: np.ndarray,
                            dZ: np.ndarray,
                            Z_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the gradients of a neural network layer given
        :param m: the input shape
        :param dV: the gradient of the induced local field
        :param dZ: the gradient of the layer output
        :param Z_prev: the layer
        :return:
        """
        db = np.expand_dims(np.sum(dZ, axis=0), axis=0) / m
        dW = Z_prev.T @ dV / m
        return db, dW

    @staticmethod
    def calculate_learning_rate(alpha: Union[Number, Callable[[int], Number]], iteration: int) -> Number:
        """
        Given the alpha parameter and iteration number, calculates the learning rate
        :param alpha: the alpha parameter of the fit method
        :param iteration: the iteration number
        :return: the learning rate
        """
        if isinstance(alpha, Callable):
            learning_rate = alpha(iteration)
        elif isinstance(alpha, Number):
            learning_rate = alpha
        else:
            raise NotImplementedError('alpha parameter can only be a number or a callable that returns a number.')
        return learning_rate

    def initialize_parameters(self):
        for layer in self.layers:
            layer.initialize_parameters()

    def forward(self, X: np.ndarray) -> Tuple[List, List]:
        """
        Training mode forward pass through the neural network
        :param X: the feature matrix
        :return: the cache for the backward pass of the backpropagation
        """
        Z = X
        V_cache, Z_cache = [], [X]
        for layer in self.layers:
            V = layer.V(Z)
            Z = layer.activation(V)
            V_cache.append(V)
            Z_cache.append(Z)
        return V_cache, Z_cache

    def backward(self, y_true: np.ndarray,
                 V_cache: List[np.ndarray],
                 Z_cache: List[np.ndarray],
                 alpha: Number) -> None:
        """
        Training mode backward pass through the neural network.
        Calculates the gradients and applies the gradient descent updates to the layer weights.
        :param y_true: the target vector
        :param V_cache: the induced local field cache
        :param Z_cache: the activation cache
        :param alpha: the learning rate
        :return: None
        """
        m = Z_cache[-1].shape[1]
        dZ = (2 / m) * np.subtract(Z_cache[-1], y_true)
        dV = self.layers[-1].activation.backward(V_cache[-1], dZ)
        db = (1 / m) * np.sum(dZ)
        dW = Z_cache[-2].T @ dV / m
        self.layers[-1].apply_gradients(alpha, db, dW)

        for i in reversed(range(0, len(self.layers) - 1)):
            m = Z_cache[i].shape[1]
            dZ = (self.layers[i + 1].W @ dV.T).T
            dV = self.layers[i].activation.backward(V_cache[i], dZ)

            db, dW = self.calculate_gradients(m, dV, dZ, Z_cache[i])
            self.layers[i].apply_gradients(alpha, db, dW)

    def step(self, X: np.ndarray, y: np.ndarray, alpha: Number) -> None:
        """
        Applies one gradient descent step on a batch of data
        :param X: the feature matrix of the batch
        :param y: the target vector of the batch
        :param alpha: the learning rate
        :return:
        """
        V_cache, Z_cache = self.forward(X)
        self.backward(y, V_cache, Z_cache, alpha)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            batch_size: int,
            epochs: int,
            alpha: Union[Number, Callable[[int], Number]],
            shuffle: bool = True,
            start_epoch: int = 0,
            cold_start: bool = False) -> 'NeuralNetwork':
        """
        Fits the neural network to the training data via gradient descent
        :param X_train: the feature matrix of training data
        :param y_train: the target vector of training data
        :param batch_size: the size of batches for each gradient descent step
        :param epochs: the number of iterations on the training data
        :param alpha: the learning rate or a callable that takes the iteration number and returns the learning rate
        :param shuffle: whether to shuffle the dataset each iteration while training
        :param cold_start: whether to reinitialize the weights before training
        :return: None
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if cold_start:
            self.initialize_parameters()
        n_batches = len(y_train) // batch_size
        for iteration in range(1 + start_epoch, 1 + start_epoch + epochs):
            learning_rate = self.calculate_learning_rate(alpha, iteration)
            indices = np.random.permutation(len(y_train)) if shuffle else np.arange(len(y_train))
            for batch in range(0, n_batches):
                batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                self.step(X_batch, y_batch, learning_rate)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Inference mode forward pass through the neural network
        :param X: the feature matrix
        :return: the predictions
        """
        Z = X
        for layer in self.layers:
            Z = layer(Z)
        return Z
