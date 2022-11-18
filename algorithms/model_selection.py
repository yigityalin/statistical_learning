import numpy as np

from .base import Model
import metrics


def k_fold(k: int,
           model: Model,
           X: np.ndarray,
           y: np.ndarray,
           shuffle: bool = True):
    pass

