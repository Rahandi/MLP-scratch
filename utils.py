import numpy as np


def to_categorical(y):
    uniq = np.unique(y)
    res = np.zeros((len(y), len(uniq)))
    for i in range(len(res)):
        res[i, y[i]] = 1.0
    return res
