import numpy as np
import random


class Layer:
    def __init__(self, n_neuron: int):
        self.n_neuron = n_neuron
        self.input_data = None
        self.output_data = None
        self.error = None

    def __call__(self, n_input: int):
        pass

    def calc_output(self):
        pass


class Input(Layer):
    def __init__(self, n_neuron: int):
        super().__init__(n_neuron)
        self.trainable = False

    def __call__(self, n_input: int):
        self.weight = np.eye(n_input, self.n_neuron)

    def calc_output(self):
        return self.input_data.copy()


class Dense(Layer):
    def __init__(self, n_neuron: int, activation: str):
        super().__init__(n_neuron)
        self.trainable = True
        self.activation = activation

    def __call__(self, n_input: int):
        self.weight = np.random.randn(n_input, self.n_neuron) * np.sqrt(2.0/n_input)

    def calc_output(self):
        return self.activation(np.matmul(self.input_data, self.weight))
