import math
from typing import Callable

import numpy as np


class Perceptron:
    def __init__(self, input_size, bias=0.25, activation_function: Callable[[float], float] = None, lr: float = 0.01):
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = bias
        self.activation_function = activation_function or (lambda x: 1 / (1 + math.exp(-x)))
        self.lr = lr

    def predict(self, inp):
        return self.activation_function(sum(inp * self.weights) + self.bias)

    def train(self, data: tuple[np.ndarray, np.ndarray], epochs: int = 200):
        input_data, expected_outputs = data
        for epoch in range(epochs):
            for x, y in zip(input_data, expected_outputs):
                err = y - self.predict(x)
                self.weights += self.lr * err * x
                self.bias += self.lr * err
