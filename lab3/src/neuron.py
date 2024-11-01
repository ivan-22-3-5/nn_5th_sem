import numpy as np


class Neuron:
    def __init__(self,
                 input_size,
                 weights=None,
                 bias=0,
                 activation_function=lambda x: x / (1 + abs(x)),
                 learning_rate=0.2):
        self.weights = np.array(weights) if weights is not None else np.random.rand(input_size)
        self.bias = bias
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def predict(self, inputs: list):
        if len(inputs) != len(self.weights):
            raise ValueError(f'Expected {len(self.weights)} inputs, got {len(inputs)}')
        return self.activation_function(self.weight(inputs))

    def weight(self, inputs: list):
        return np.dot(self.weights, inputs) + self.bias

    def __repr__(self):
        return f'Neuron({self.weights}, input_size={len(self.weights)}, bias={self.bias})'
