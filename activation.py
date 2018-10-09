import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1-(x**2)
    return np.tanh(x)


def linear(x, derivative=False):
    if derivative:
        return 1
    return x


def relu(x, derivative=False):
    result = x.copy()
    if derivative:
        result[result > 0] = 1.0
        result[result < 0] = 0.0
        return result

    result[result < 0] = 0.0
    return result


def arctan(x, derivative=False):
    if derivative:
        return np.cos(x)**2
    return np.arctan(x)


def step(x, derivative=False):
    if derivative:
        for a in range(len(x)):
            for b in range(len(x[a])):
                if x[a][b] > 0:
                    x[a][b] = 1
                else:
                    x[a][b] = 0
        return x
    for a in range(len(x)):
        for b in range(len(x[a])):
            if x[a][b] > 0:
                x[a][b] = 1
            else:
                x[a][b] = 0
    return x


def squash(x, derivative=False):
    if derivative:
        for a in range(len(x)):
            for b in range(len(x[a])):
                if x[a][b] > 0:
                    x[a][b] = x[a][b] / (1+x[a][b])
                else:
                    x[a][b] = x[a][b] / (1-x[a][b])
        return x
    for a in range(len(x)):
        for b in range(len(x[a])):
            x[a][b] = x[a][b] / (1+abs(x[a][b]))
    return x


def gaussian(x, derivative=False):
    if derivative:
        for a in range(len(x)):
            for b in range(len(x[a])):
                x[a][b] = -2*x[a][b] * np.exp(-x[a][b]**2)
    for a in range(len(x)):
        for b in range(len(x[a])):
            x[a][b] = np.exp(-x[a][b]**2)
    return x


func = {
    'sigmoid': sigmoid,
    'relu': relu,
    'linear': linear
}
