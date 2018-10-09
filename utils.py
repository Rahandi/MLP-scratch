import numpy as np


def to_categorical(y: np.ndarray):
    uniq = np.unique(y)
    res = np.zeros((len(y), len(uniq)))
    for i in range(len(res)):
        res[i, y[i]] = 1.0
    return res


def normalize(x: np.ndarray):
    """Scale to 0-1
    """
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def standardize(x: np.ndarray):
    """Scale to zero mean unit variance
    """
    return (x - x.mean(axis=0)) / np.std(x, axis=0)
