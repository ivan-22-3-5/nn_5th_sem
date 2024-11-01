import numpy as np
from scipy.misc import derivative

from src.neuron import Neuron


class Network:
    def __init__(self, *layer_sizes, test_mode: bool = False):
        if test_mode:
            self.layers = [
                [Neuron(2, [0.1, 0.3]), Neuron(2, [-0.2, 0.4]), Neuron(2, [0.3, -0.25]), Neuron(2, [0.15, 0.1])],
                [Neuron(4, [0.2, -0.2, -0.1, 0.3]), Neuron(4, [0.1, 0.3, -0.4, 0.5]), Neuron(4, [0.5, 0.4, 0.2, -0.1])],
                [Neuron(3, [-0.15, 0.3, 0.4]), Neuron(3, [0.5, 0.25, -0.2])],
            ]
            for layer in self.layers:
                for neuron in layer:
                    neuron.activation_function = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.layers = []
            for number_of_neurons, input_size in zip(layer_sizes[1:], layer_sizes):
                self.layers.append([Neuron(input_size) for _ in range(number_of_neurons)])

    def predict(self, inputs):
        for layer in self.layers:
            inputs = [neuron.predict(inputs) for neuron in layer]
        return inputs

    def _get_internal_outputs(self, inputs) -> list[list[float]]:
        outputs = []
        for layer in self.layers:
            inputs = [neuron.predict(inputs) for neuron in layer]
            outputs.append(inputs)
        return outputs

    def _get_internal_errors(self, inputs, expected_outputs):
        outputs = self._get_internal_outputs(inputs)
        errors = [[]]
        for neuron, output, expected_output in zip(self.layers[-1], outputs[-1], expected_outputs):
            errors[0].append(
                (expected_output - output) * derivative(neuron.activation_function, output, dx=0.001))
        for layer, prev_layer, inp in zip(reversed(self.layers[:-1]), reversed(self.layers[1:]),
                                          reversed(([inputs] + outputs)[:-2])):
            errors.append([])
            for index, neuron in enumerate(layer):
                term = sum(prev_layer[k].weights[index] * errors[-2][k] for k in range(len(prev_layer)))
                errors[-1].append(term * derivative(neuron.activation_function, neuron.weight(inp)))
        return errors
