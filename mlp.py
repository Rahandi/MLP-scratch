from layer import Layer

class MLP:
    def __init__(self, ):
        self.layer = []
        self.loss = 0

    def add(self, n_neuron, n_input, activation):
        new_layer = Layer('hidden', activation, n_neuron)
        self.layer.append(new_layer)