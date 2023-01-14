from collections import defaultdict
from numbers import Number
from typing import Callable, Tuple, Union

from tqdm import tqdm
import numpy as np

from . import utils
from .base import Model
from .metrics import mse
from .model_selection import DEFAULT_METRICS


def check_dimensions(X=None, y=None) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
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


class LinearRegression(Model):
    """
    Linear regression model that fits the parameters using gradient descent.
    Only MSE is supported for gradient descent updates.

    Attributes:
        b : np.ndarray
            the bias vector of the model
        W : np.ndarray
            the weight matrix of the model
        alpha : Union[Number, Callable[[int], Number]]
            the learning rate of the model
        lambda_ : float
            the regularization parameter of the model
    """
    def __init__(self, alpha: Union[Number, Callable[[int], Number]] = 1e-2, lambda_: float = 0):
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
    def b(self) -> np.ndarray:
        """
        The y-intercept of the model
        :return: the y-intercept
        """
        return self._b

    @property
    def W(self) -> np.ndarray:
        """
        The weight matrix of the model
        :return: the weight matrix
        """
        return self._W

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Wraps the model's predict method
        :param X: the feature matrix
        :return: predictions
        """
        return self.predict(X)

    def __repr__(self) -> str:
        """
        Returns the initialization signature of the instance
        :return: the string representation
        """
        return f'LinearRegression(alpha={self.alpha}, lambda_={self.lambda_})'

    def __str__(self) -> str:
        """
        Calls the repr method of the class
        :return: the string representation
        """
        return repr(self)

    def initialize_parameters(self, in_features: int):
        self._b, self._W = utils.initialize_parameters(b_shape=(1, 1), W_shape=(in_features, 1))

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_valid,
            y_valid,
            epochs: int = None,
            batch_size: int = 32,
            min_delta: float = 1e-7,
            patience: int = 50,
            shuffle: bool = True,
            cold_start: bool = False) -> dict:
        """
        Calculates the weights and bias of the model using the gradient descent algorithm
        :param X: the feature matrix
        :param y: the target vector
        :param max_iter: maximum number of iterations
        :param tolerance: the tolerance for MSE loss below which the parameters are acceptable
        :param cold_start: whether to reinitialize the weights before training
        :return: the regressor itself
        """
        X = np.asarray(X)
        y = np.asarray(y)

        X, y = check_dimensions(X, y)

        if cold_start or self.b is None or self.W is None:
            self.initialize_parameters(X.shape[-1])

        n_batches = len(X) // batch_size
        history = defaultdict(list)

        n_no_improvement = 0
        for iteration in (progress_bar := tqdm(range(epochs))):
            alpha = self._get_alpha(iteration)
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_average_metrics = defaultdict(list)
            for batch in range(n_batches):
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                grad_b, grad_W = self._calculate_gradients(X_batch, y_batch)
                y_batch_pred = self.predict(X_batch)
                for metric, fn in DEFAULT_METRICS.items():
                    batch_average_metrics[metric].append(fn(y_batch, y_batch_pred))
                self._b -= alpha * grad_b
                self._W -= alpha * grad_W
            train_avg_losses = {metric: np.mean(batch_average_metrics[metric])
                                for metric in DEFAULT_METRICS.keys()}
            valid_avg_losses = {metric: np.mean(fn(y_valid, self.predict(X_valid)))
                                for metric, fn in DEFAULT_METRICS.items()}

            for metric in DEFAULT_METRICS.keys():
                history[f'train_{metric}'].append(train_avg_losses[metric])
                history[f'valid_{metric}'].append(valid_avg_losses[metric])

            progress_bar.set_description_str(f'alpha={alpha}, lambda={self.lambda_}, batch_size={batch_size}')
            progress_bar.set_postfix_str(f'train_mse={train_avg_losses["MSE"]:.7f}, '
                                         f'valid_mse={valid_avg_losses["MSE"]:.7f}')

            if iteration > 2 and history['valid_MSE'][-2] - history['valid_MSE'][-1] < min_delta:
                n_no_improvement += 1
                if n_no_improvement > patience:
                    break
            else:
                n_no_improvement = 0
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.b is None or self.W is None:
            raise RuntimeError('The model is not fit.')
        X = np.asarray(X)
        X, _ = check_dimensions(X)
        return self.b + X @ self.W

    def _get_alpha(self, iteration: int) -> float:
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

    def _calculate_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradients for MSE loss with current bias and weights
        :param X: the feature matrix
        :param y: the target vector
        :return: the gradients of the bias and the weights
        """
        y_pred = self.b + X @ self.W
        grad_b = 2 * np.sum(y_pred - y) / len(y)
        grad_W = 2 * (X.T @ (y_pred - y) + self.lambda_ * np.sign(self.W)) / len(y)
        return grad_b, grad_W
