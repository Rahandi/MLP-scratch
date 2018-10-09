import pandas as pd
import numpy as np
from utils import to_categorical, normalize, standardize
from mlp import MultiLayerPerceptron
from layer import Input, Dense


def main():
    data = pd.read_csv('iris.csv')
    x_train = data.drop(['target'], axis=1)
    y_train = data['target']

    x_train = pd.get_dummies(x_train)
    target_map = dict()
    if y_train.dtype == 'object':
        target_map = {val: i for (i, val) in enumerate(np.unique(y_train))}
        # print(target_map)
        y_train = y_train.map(target_map)
        y_train = to_categorical(y_train)
        # print(y_train[:5])
    x_train = x_train.values
    # x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
    # x_train = (x_train - x_train.mean(axis=0)) / np.std(x_train, axis=0)
    x_train = standardize(x_train)
    mlp = MultiLayerPerceptron()
    mlp.add(Input(x_train.shape[1]))
    mlp.add(Dense(32, activation='relu'))
    mlp.add(Dense(3, activation='sigmoid'))
    mlp.build()
    mlp.fit(x_train, y_train, epoch=100, lr=0.1)


if __name__ == '__main__':
    main()
