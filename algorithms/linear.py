from numbers import Number
from typing import Callable

import numpy as np

from . import utils


class LinearRegression:
    """
    Linear regression model that fits the parameters using gradient descent.
    Only MSE is supported for gradient descent updates.

    Attributes:
        b : np.ndarray
            the bias vector of the model
        W : np.ndarray
            the weight vector of the model
        alpha : Number | Callable[[int], int]
            the learning rate of the model
        lambda_ : float
            the regularization parameter of the model
    """
    def __init__(self, alpha: Number | Callable[[int], int] = 1e-2, lambda_: float = 0, normalize_features: bool = True):
        """
        The init method of the LinearRegression model
        :param alpha: the learning rate
        :param lambda_: the regularization constant
        :param normalize_features:
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self._b = None
        self._W = None
        if self.lambda_ < 0:
            raise ValueError('lambda must satisfy >= 0')

    @property
    def b(self):
        """
        The y-intercept of the model
        :return: the y-intercept
        """
        return self._b

    @property
    def W(self):
        """
        The weight vector of the model
        :return: the weight vector
        """
        return self._W

    def __repr__(self) -> str:
        return f'LinearRegression(alpha={self.alpha}, lambda_={self.lambda_})'

    def __str__(self) -> str:
        return repr(self)

    def fit(self, X: np.ndarray, y: np.ndarray = None, max_iter: int = None, tolerance: float = 1e-14):
        """
        Calculates the weights and bias of the model using the gradient descent algorithm
        :param X: the feature matrix
        :param y: the target vector
        :param max_iter: maximum number of iterations
        :param tolerance: the tolerance for MSE loss below which the parameters are acceptable
        :return: the regressor itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError('The number of dimensions of the feature matrix has to be 2.')
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        elif y.ndim != 2:
            raise ValueError('The shape of the target matrix has to be (n, 1) or (n,),'
                             ' where n is the number of the training samples')

        self._b, self._W = utils.initialize_parameters(b_shape=(1, 1), W_shape=(X.shape[-1], 1))
        iteration, grad_b, grad_W = 0, np.inf, np.inf
        if max_iter is None:
            max_iter = np.inf
        while iteration < max_iter and not np.isclose(grad_b, 0, atol=tolerance).all() \
                and not np.isclose(grad_W, 0, atol=tolerance).all():
            alpha = self._get_alpha(iteration)
            grad_b, grad_W = self._calculate_gradients(X, y)
            self._b -= alpha * grad_b
            self._W -= alpha * grad_W
            iteration += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for a given feature matrix
        :param X: the feature matrix
        :return: predictions
        """
        if self.b is None or self.W is None:
            raise RuntimeError('The model is not fit.')
        return self.b + X @ self.W

    def _get_alpha(self, iteration):
        """
        Calculates the learning rate.
        Returns the alpha parameter of the model if it is a float.
        Calls the alpha of the model with the current iteration number if it is a callable,
        :param iteration: the iteration number
        :return: learning rate
        """
        if isinstance(self.alpha, Number):
            alpha = self.alpha
        elif isinstance(self.alpha, Callable):
            alpha = self.alpha(iteration)
        else:
            raise ValueError('The alpha parameters must be of type float or Callable[[int], int]')
        return alpha

    def _calculate_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradients for MSE loss with current bias and weights
        :param X: the feature matrix
        :param y: the target vector
        :return: the gradients of the bias and the weights
        """
        y_pred = self.b + X @ self.W
        grad_b = 2 * np.sum(y_pred - y) / len(y)
        grad_W = 2 * (X.T @ (y_pred - y) + self.lambda_ * self.W) / len(y)
        return grad_b, grad_W
