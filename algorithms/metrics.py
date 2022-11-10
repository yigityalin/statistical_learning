import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.square(y_true - y_pred)) / len(y_true)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.divide(np.abs(y_true - y_pred), y_true)) / len(y_pred)
