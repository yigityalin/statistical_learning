import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error given two vectors
    :param y_true: the true targets
    :param y_pred: the predicted targets
    :return: the mean squared error
    """
    return np.sum(np.square(y_true - y_pred)) / len(y_true)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Calculates the mean absolute error given two vectors
        :param y_true: the true targets
        :param y_pred: the predicted targets
        :return: the mean absolute error
        """
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Calculates the mean absolute percentage error given two vectors
        :param y_true: the true targets
        :param y_pred: the predicted targets
        :return: the mean absolute percentage error
    """
    return np.sum(np.divide(np.abs(y_true - y_pred), y_true)) / len(y_pred)
