import numpy as np


class Neuron:
    def __init__(self,
                 input_size,
                 weights=None,
                 bias=0,
                 activation_function=lambda x: x / (1 + abs(x)),
                 learning_rate=0.2):
        self.weights = np.array(weights) if weights is not None else np.random.rand(input_size) - 0.5
        self.bias = bias
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def predict(self, inp: list):
        if len(inp) != len(self.weights):
            raise ValueError(f'Expected {len(self.weights)} inputs, got {len(inp)}')
        return self.activation_function(self.weight(inp))

    def weight(self, inp: list):
        return np.dot(self.weights, inp) + self.bias

    def update_weights(self, inp: list, err: float):
        self.weights += self.learning_rate * err * np.array(inp)

    def __repr__(self):
        return (f'Neuron({self.weights}, '
                f'input_size={len(self.weights)}, '
                f'bias={self.bias}, '
                f'learning_rate={self.learning_rate}'
                f'weights={self.weights})')
