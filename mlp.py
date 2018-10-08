import numpy as np
from layer import Layer


class MLP:

    def __init__(self):
        """Multi Layer Perceptron 
        """
        self.layers = []
        self.loss = None
        self.metric = None
        self.x_train = None
        self.y_train = None
        self.lr = None

    def add(self, layer: Layer):
        """Add layer to MLP

        Parameters
        ----------
        layer : Layer
            Instance of Layer
        """
        self.layers.append(layer)

    def forward(self):
        """Loop over MLP's layer from input to output layer
        """

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].output_data = self.layers[i].calc_output()
            else:
                self.layers[i].output_data = self.layers[i].activation(self.layers[i].calc_output())
            if i != len(self.layers) - 1:
                self.layers[i+1].input_data = self.layers[i].output_data.copy()

    def backward(self):
        for i in reversed(range(len(self.layers))):
            if self.layers[i].trainable:
                if i == len(self.layers) - 1:
                    self.layers[i].error = self.y_train - self.layers[i].activation(self.layers[i].output_data)
                else:
                    self.layers[i].error = np.dot(self.layers[i+1].weight, self.layers[i+1].error)
        for i in range(len(self.layers)):
            if self.layers[i].trainable:
                dw = self.layers[i].input_data.T.dot(
                    self.layers[i].error * self.layers[i].activation(self.layers[i].output_data, derivative=True))
                self.layers[i].weight += self.lr * dw

    def build(self, loss, metrics):
        for i in range(1, len(self.layers)):
            self.layers[i](self.layers[i-1].n_neuron)

    def fit(self, x_train, y_train, epoch):
        for e in range(epoch):
            losses = []
            for (row_x, row_y) in zip(x_train, y_train):
                self.x_train = np.atleast_2d(row_x)
                self.y_train = np.atleast_2d(row_y)

                self.forward()
                self.backward()
                loss = float(self.layers[-1].activation(self.layers[-1].output_data).squeeze())
                losses.append(loss)
