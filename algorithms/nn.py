from numbers import Number
from typing import Callable, Iterable, List, Tuple, Type, Union

import numpy as np

from . import utils
from .activations import Activation


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
        self._b, self._W = utils.initialize_parameters(b_shape=(1, out_features), W_shape=(in_features, out_features))

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


class NeuralNetwork:
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
        Inference mode forward pass through the neural network
        :param X: the feature matrix
        :return: the predictions
        """
        Z = X
        for layer in self.layers:
            Z = layer(Z)
        return Z

    @staticmethod
    def calculate_gradients(m: int,
                            dV: np.ndarray,
                            dZ: np.ndarray,
                            Z_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the gradients of a neural network layer given
        :param m:
        :param dV:
        :param dZ:
        :param Z_prev:
        :return:
        """
        db = np.expand_dims(np.sum(dZ, axis=1), axis=1) / m
        dW = dV @ Z_prev.T / m
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

    def backward(self, y_true: np.ndarray, V_cache: List[np.ndarray], Z_cache: List[np.ndarray], alpha: Number) -> None:
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
        dZ = (2 / m) * np.subtract(Z_cache[-1], np.array(y_true))
        dV = self.layers[-1].activation.backward(V_cache[-1], dZ)
        db = (1 / m) * np.sum(dZ)
        dW = Z_cache[-2].T @ dV / m
        self.layers[-1].apply_gradients(alpha, db, dW)

        for i in reversed(range(0, len(self.layers) - 1)):
            m = Z_cache[i].shape[1]
            dZ = self.layers[i + 1].W.T @ dV
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
            X_validation: np.ndarray,
            y_validation: np.ndarray) -> None:
        """
        Fits the neural network to the training data via gradient descent
        :param X_train: the feature matrix of training data
        :param y_train: the target vector of training data
        :param batch_size: the size of batches for each gradient descent step
        :param epochs: the number of iterations on the training data
        :param alpha: the learning rate or a callable that takes the iteration number and returns the learning rate
        :param X_validation: the feature matrix of validation data
        :param y_validation: the target vector of validation data
        :return: None
        """
        n_batches = len(y_train) // batch_size
        for iteration in range(1, epochs + 1):
            learning_rate = self.calculate_learning_rate(alpha, iteration)
            for batch in range(0, n_batches):
                X_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                self.step(X_batch, y_batch, learning_rate)

        # TODO: Calculate accuracy on validation dataset