from collections import namedtuple
from typing import Callable, Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import Model
from . import metrics as m


DEFAULT_METRICS = {
    'MSE': m.mse,
    'MAE': m.mae,
    'MAPE': m.mape,
    'R2': m.r2,
}

TargetVectorPair = namedtuple('TargetVectorPair', ['y_true', 'y_pred'])
TrainValidationPair = namedtuple('TrainValidationPair', ['train', 'validation'])


def create_train_validation_pair(y_train: np.ndarray,
                                 y_train_pred: np.ndarray,
                                 y_valid: np.ndarray,
                                 y_valid_pred: np.ndarray) -> TrainValidationPair:
    """
    Create a train-validation pair
    :param y_train: the true train target vector
    :param y_train_pred: the train target vector predictions
    :param y_valid: the true validation target vector
    :param y_valid_pred: the valiation target vector predictions
    :return: the train-validation pair with given inputs
    """
    train_targets = TargetVectorPair(y_true=y_train, y_pred=y_train_pred)
    valid_targets = TargetVectorPair(y_true=y_valid, y_pred=y_valid_pred)
    return TrainValidationPair(train=train_targets, validation=valid_targets)


class KFold:
    """
    K-Fold Cross Validation class

    Attributes:
        raw_results: List[TrainValidationPair[TargetVectorPair[np.ndarray, np.ndarray],
                                              TargetVectorPair[np.ndarray, np.ndarray]]]
            true and predicted target vectors for each fold
        results: List[TrainValidationPair[Dict[str, float], Dict[str, float]]]
            metrics for train and validation datasets
        models: List[Model]
            list of trained models for each fold
    """
    def __init__(self,
                 k: int,
                 model: Model,
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None):
        """
        Initialize the KFold class
        :param k: the number of the folds
        :param model: the model to cross validate
        :param metrics: the metrics used to cross validate
        """
        self.k = k
        self.model = model
        self.metrics = metrics if metrics else DEFAULT_METRICS
        self._raw_results = None
        self._results = None
        self._models = None

    def __repr__(self) -> str:
        """
        The string representation of the KFold instance
        :return: the string representation
        """
        return f'KFold(k={self.k}, model={self.model})'

    def __str__(self) -> str:
        """
        The string representation of the KFold instance
        :return: the string representation
        """
        return repr(self)

    @property
    def raw_results(self) -> List[TrainValidationPair[TargetVectorPair[np.ndarray, np.ndarray],
                                                      TargetVectorPair[np.ndarray, np.ndarray]]]:
        """
        The results of the k-fold cross validation
        :return: the results list
        """
        if self._raw_results is None:
            raise RuntimeError('Please run cv first to obtain the results')
        return self._raw_results

    @property
    def results(self) -> List[TrainValidationPair[Dict[str, float], Dict[str, float]]]:
        """
        The results of the k-fold cross validation
        :return: the results list
        """
        if self._results is None:
            self._results = self.process_raw_results()
        return self._results

    @property
    def models(self) -> Dict[str, Dict[str, float]]:
        """
        The mean and std of the k-fold cross validation results
        :return: the results list
        """
        if self._models is None:
            raise RuntimeError('Please run cv first to obtain the results')
        return self._models

    def get_scores(self, as_dataframe: bool = False) -> Union[Tuple[dict[str, dict[str, float]]],
                                                              Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get the scores using the results of the cv
        :param as_dataframe: whether to return pandas dataframes or nested dictionaries
        :return: the mean and standard deviation of the metric scores
        """
        train_scores, test_scores = dict(), dict()
        for metric in self.metrics.keys():
            train_mean = np.mean([fold_results.train[metric] for fold_results in self.results])
            test_mean = np.mean([fold_results.train[metric] for fold_results in self.results])

            train_std = np.sqrt(np.mean([(fold_results.train[metric] - train_mean) ** 2
                                         for fold_results in self.results]))
            test_std = np.sqrt(np.mean([(fold_results.validation[metric] - train_mean) ** 2
                                        for fold_results in self.results]))

            train_scores[metric] = dict(mean=train_mean, std=train_std)
            test_scores[metric] = dict(mean=test_mean, std=test_std)

        if as_dataframe:
            train_scores = pd.DataFrame.from_dict(train_scores)
            test_scores = pd.DataFrame.from_dict(test_scores)
        return train_scores, test_scores

    def get_best_model(self, n: int = 1,
                       metric: Callable[[np.ndarray, np.ndarray], float] = m.r2,
                       objective: Literal['min', 'max'] = 'max'):
        """
        Gets the best n models
        :param n: the number of the models
        :param metric: the metric according to which the best models will be determined
        :param objective: whether to minimize or maximize the metric
        :return:
        """
        if not 0 < n < self.k:
            raise ValueError('n must be between 0 and k.')
        indices = np.argsort([metric(*fold_results.validation) for fold_results in self.raw_results])
        if objective == 'min':
            indices = indices[:n]
        elif objective == 'max':
            indices = list(reversed(indices))[:n]
        return [self.models[index] for index in indices]

    def _reset_results(self) -> None:
        """
        Resets the KFold instance by setting the results and models to None,
        and the raw results to an empty list
        :return: None
        """
        self._raw_results = []
        self._results = None
        self._models = []

    def cv(self,
           X: np.ndarray,
           y: np.ndarray,
           shuffle: bool = True,
           **fit_params) -> 'KFold':
        """
        Apply k-fold cross validation with given dataset
        :param X: the feature matrix
        :param y: the target vector
        :param shuffle: whether to shuffle the data before cross validation
        :param fit_params: the keyword arguments to pass to the model's fit method
        :return: the KFold instance itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self._reset_results()
        indices = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
        fold_size = len(y) // self.k
        for fold in range(self.k):
            valid_indices = indices[fold * fold_size: (fold + 1) * fold_size]
            train_indices = indices[~np.isin(indices, valid_indices)]

            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            self.model.fit(X_train, y_train, cold_start=True, **fit_params)
            y_train_pred = self.model.predict(X_train)
            y_valid_pred = self.model.predict(X_valid)

            result = create_train_validation_pair(y_train, y_train_pred, y_valid, y_valid_pred)
            self._raw_results.append(result)
            self._models.append(self.model.copy())
        return self

    def process_raw_results(self) -> List[TrainValidationPair[Dict[str, float], Dict[str, float]]]:
        """
        Processes the raw_results and returns the metric outputs for train and validation datasets
        :return: the list of TrainValidationPairs containing the metrics for each fold
        """
        results = []
        for tv_pair in self.raw_results:
            scores_train = {metric: metric_fn(*tv_pair.train)
                            for metric, metric_fn in self.metrics.items()}
            scores_valid = {metric: metric_fn(*tv_pair.validation)
                            for metric, metric_fn in self.metrics.items()}
            scores = TrainValidationPair(scores_train, scores_valid)
            results.append(scores)
        return results
