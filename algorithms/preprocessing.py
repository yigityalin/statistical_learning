from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd

import config


def load_dataset(path: Union[str, Path] = config.DATASET_PATH) -> pd.DataFrame:
    """
    Loads  dataset
    :param path: the file path of the dataset
    :return: the dataset
    """
    return pd.read_csv(path)


def split_dataset(data: pd.DataFrame,
                  target: str = config.TARGET,
                  test_split: float = 0.1,
                  seed: int = 42
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset
    :param data: the dataset
    :param target: the target column
    :param test_split: the fraction of the test data
    :param seed: the random seed
    :return: the train and test datasets
    """
    test_data = data.sample(frac=test_split, random_state=seed)
    train_data = data.drop(test_data.index)
    return train_data.drop(target, axis=1), train_data[[target]], \
           test_data.drop(target, axis=1), test_data[[target]],


def encode_categorical(data: pd.DataFrame,
                       columns: Union[Iterable[str], None] = None,
                       drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encodes the categorical data
    :param data: the dataset
    :param columns: the columns to encode. The columns are inferred if set to None
    :param drop_first: whether to drop the first categorical level
    :return:
    """
    if columns is not None:
        columns = list(columns)
    else:
        if config.TARGET in data.columns:
            data = data.drop(config.TARGET, axis=1)
        columns = list(data.select_dtypes(include=object).columns)
    return pd.get_dummies(data, columns=columns, drop_first=drop_first)


def normalize_columns(data: pd.DataFrame,
                      data_test: Union[None, pd.DataFrame] = None,
                      columns: Union[Iterable[str], None] = None
                      ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Union[None, pd.DataFrame]]]:
    """
    Normalize the features
    :param data: the train dataset
    :param data_test: the test dataset
    :param columns: the columns to normalize
    :return: the normalized feature matrix
    """
    data = data.copy()
    if columns is not None:
        columns = list(columns)
    else:
        if config.TARGET in data.columns:
            data = data.drop(config.TARGET, axis=1)
        columns = list(data.select_dtypes(exclude=object).columns)
    mean = data[columns].mean(axis=0)
    std = data[columns].std(axis=0)
    data[columns] = (data[columns] - mean) / std
    if data_test is not None:
        data_test[columns] = (data_test[columns] - mean) / std
    return data, data_test


def load_and_preprocess_dataset(path: Union[str, Path] = config.DATASET_PATH,
                                target: str = config.TARGET,
                                test_split: float = 0.1,
                                seed: int = 42,
                                normalize_features: bool = True,
                                normalize_target: bool = True,
                                encode: bool = True,
                                drop_first: bool = True
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param path: the dataset filepath
    :param target: the target column
    :param test_split: the fraction of the data split for the test
    :param seed: the random seed
    :param normalize_features: whether to normalize features
    :param normalize_target: whether to normalize target
    :param encode: whether to one-hot encode the variables
    :param drop_first: whether to drop one feature column for each categorical variable
    :return: the dataset
    """
    data = load_dataset(path)
    if test_split == 0:
        X_train, y_train, X_test, y_test = data.drop(target, axis=1), data[[target]], None, None
    else:
        X_train, y_train, X_test, y_test = split_dataset(data, target, test_split, seed)
    if normalize_features:
        X_train, X_test = normalize_columns(X_train, X_test)
    if normalize_target:
        y_train, y_test = normalize_columns(y_train, y_test, columns=[config.TARGET])
    if encode:
        X_train = encode_categorical(X_train, drop_first=drop_first)
        X_test = encode_categorical(X_test, drop_first=drop_first)
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        difference = list(train_features - test_features)
        X_test[difference] = np.zeros((len(X_test), len(difference)))
    return X_train, y_train, X_test, y_test
