import numpy as np
import random

class Layer:
    def __init__(self, layer_type, activation, n_neuron):
        self.n_neuron = n_neuron
        self.activation = activation
    
    def __call__(self, n_input):
        self.weight = np.random.randn(n_input,self.n_neuron) * np.sqrt(2.0/n_input)

    def output(self, data_input):
        return self.activation(np.matmul(data_input, self.weight))