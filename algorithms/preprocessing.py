from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

import config


def load_dataset(path: str | Path = config.DATASET_PATH) -> pd.DataFrame:
    """
    Loads  dataset
    :param path: the file path of the dataset
    :return: the dataset
    """
    return pd.read_csv(path)


def split_dataset(data: pd.DataFrame,
                  target: str = config.TARGET,
                  test_split: float = 0.2,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return train_data.drop(target, axis=1), train_data[[target]],\
           test_data.drop(target, axis=1), test_data[[target]],


def encode_categorical(data: pd.DataFrame,
                       columns: Iterable[str] | None = None,
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
        columns = list(data.drop(config.TARGET, axis=1).select_dtypes(include=object).columns)
    return pd.get_dummies(data, columns=columns, drop_first=drop_first)


def normalize_features(data: pd.DataFrame,
                       columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Normalize the features
    :param data: the dataset
    :param columns: the columns to normalize
    :return: the normalized feature matrix
    """
    data = data.copy()
    if columns is not None:
        columns = list(columns)
    else:
        columns = list(data.drop(config.TARGET, axis=1).select_dtypes(exclude=object).columns)
    data[columns] = (data[columns] - data[columns].mean(axis=0)) / data[columns].std(axis=0)
    return data


def load_and_preprocess_dataset(path: str | Path = config.DATASET_PATH,
                                target: str = config.TARGET,
                                test_split: float = 0.2,
                                seed: int = 42,
                                normalize: bool = True,
                                encode: bool = True,
                                drop_first: bool = True) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_dataset(path)
    if normalize:
        data = normalize_features(data)
    if encode:
        data = encode_categorical(data, drop_first=drop_first)
    if test_split == 0:
        return data
    else:
        return split_dataset(data, target, test_split, seed)
