import numpy as np


class Neuron:
    def __init__(self, input_size, bias=None, activation_function=None):
        self.weights = np.random.rand(input_size)
        self.bias = bias or 0.5
        self.activation_function = activation_function or (lambda x: x)

    def predict(self, inputs: list):
        if len(inputs) != len(self.weights):
            raise ValueError(f'Expected {len(self.weights)} inputs, got {len(inputs)}')
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(total)
