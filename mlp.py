from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from tqdm import trange


class MultiLayerPerceptron:
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

    def forward(self, input_data: np.ndarray):
        """Loop over MLP's layer from input to output layer

        Parameters
        ----------
        input_data : np.ndarray
            Training data's feature with shape (1, n)
        """

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].output_data = self.layers[i].calc_output(input_data)
            else:
                self.layers[i].output_data = self.layers[i].activation(self.layers[i].calc_output(input_data))
            input_data = self.layers[i].output_data

    def backward(self, target: np.ndarray):
        """Backward propagate error

        Parameters
        ----------
        target : np.ndarray
            Training data's target with shape (1,1)
        """

        for i in reversed(range(len(self.layers))):
            if self.layers[i].trainable:
                if i == len(self.layers) - 1:
                    self.layers[i].error = target - self.layers[i].output_data
                    self.layers[i].error = self.layers[i].error * self.layers[i].activation(self.layers[i].output_data)
                    self.layers[i].error = self.layers[i].error.squeeze()
                else:
                    self.layers[i].error = np.dot(self.layers[i + 1].weight, self.layers[i + 1].error.T)
                    self.layers[i].error = self.layers[i].error.squeeze()
                self.layers[i].error = np.atleast_2d(self.layers[i].error)

        for i in range(1, len(self.layers)):
            dw = (
                self.layers[i - 1]
                .activation(self.layers[i - 1].output_data)
                .T.dot(self.layers[i].error * self.layers[i].activation(self.layers[i].output_data, derivative=True))
            )
            self.layers[i].weight += self.lr * dw

    def build(self):
        """Initialize weight on layers
        """
        for i in range(1, len(self.layers)):
            self.layers[i](self.layers[i - 1].n_neuron)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epoch: np.ndarray,
        lr: float,
        validation_data=Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, List]:
        """Train model on input data

        Parameters
        ----------
        x_train : np.ndarray
            Training feature
        y_train : np.ndarray
            Training target
        epoch : np.ndarray
            Training epoch
        lr : float
            Learning rate

        Returns
        -------
        Dict[str, List]
            Contains loss and metrics score history
        """
        self.hist = {"loss": [], "metric": []}
        self.lr = lr
        data_x = x_train.copy()
        data_y = y_train.copy()
        for e in range(epoch):
            np.random.seed(e)
            idx = np.array([i for i in range(len(x_train))])
            np.random.shuffle(idx)
            x_train = data_x[idx]
            y_train = data_y[idx]
            losses = []
            with trange(len(y_train)) as t:
                for i in t:
                    feature = np.atleast_2d(x_train[i])
                    target = np.atleast_2d(y_train[i])

                    self.forward(feature)
                    self.backward(target)
                    loss = (target - self.layers[-1].activation(self.layers[-1].output_data).squeeze()) ** 2
                    losses.append(np.average(loss))

                    t.set_description("EPOCH {}".format(e + 1))
                    if validation_data is not None:
                        x_valid = validation_data[0]
                        y_valid = validation_data[1]
                        t.set_postfix(loss=np.average(losses), metric=self.evaluate(x_valid, y_valid))
                    else:
                        t.set_postfix(loss=np.average(losses), metric=self.evaluate(x_train, y_train))
            self.hist["loss"].append(np.average(losses))
            self.hist["metric"].append(self.evaluate(x_train, y_train))
        return self.hist

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict input data

        Parameters
        ----------
        x_data : np.ndarray


        Returns
        -------
        np.ndarray
            Prediction result (one hot encoded or real values)
        """

        result = []
        for data in x_data:
            self.forward(data)
            res = self.layers[-1].output_data.squeeze()
            result.append(res)
        return np.array(result)

    def evaluate(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Evaluate prediction

        Parameters
        ----------
        x_data : np.ndarray
            Feature
        y_data : np.ndarray
            Target

        Returns
        -------
        float
            Metrics score (MSE or accuracy)
        """

        result = self.predict(x_data)

        # Regression
        if self.layers[-1].n_neuron == 1:
            metric = np.average((result - y_data) ** 2)
            return metric
        # Classification
        else:
            result = np.argmax(result, axis=1)
            y_data = np.argmax(y_data, axis=1)
            metric = np.average(np.equal(result, y_data))
            return metric

    def draw(self):
        x = range(1, len(self.hist["loss"]) + 1)
        y = self.hist["loss"]
        plt.plot(x, y)
        plt.show()
