from scipy.misc import derivative

from src.neuron import Neuron


# noinspection PyShadowingBuiltins
class Network:
    def __init__(self, *layer_sizes):
        if len(layer_sizes) < 3:
            raise ValueError('Network must have at least 3 layers')
        self.layers = []
        for number_of_neurons, input_size in zip(layer_sizes[1:], layer_sizes):
            self.layers.append([Neuron(input_size) for _ in range(number_of_neurons)])

    def predict(self, input: list[float]):
        for layer in self.layers:
            input = [neuron.predict(input) for neuron in layer]
        return input

    def _calculate_outputs(self, input: list[float]) -> list[list[float]]:
        outputs = [input]
        for layer in self.layers:
            outputs.append([neuron.predict(outputs[-1]) for neuron in layer])
        return outputs

    def _calculate_errors(self, input, expected_output):
        iternal_outputs = self._calculate_outputs(input)
        errors = [[]]

        for neuron, actual, expected in zip(self.layers[-1], iternal_outputs[-1], expected_output):
            pre_activation = neuron.weight(iternal_outputs[-2])
            der = derivative(neuron.activation_function, pre_activation, dx=0.001)
            errors[-1].append((expected - actual) * der)

        for cur_layer, next_layer, cur_layer_input in zip(reversed(self.layers[:-1]),
                                                          reversed(self.layers[1:]),
                                                          reversed(iternal_outputs[:-2])):
            errors.append([])
            for index, neuron in enumerate(cur_layer):
                term = sum(next_layer[k].weights[index] * errors[-2][k] for k in range(len(next_layer)))
                der = derivative(neuron.activation_function, neuron.weight(cur_layer_input), dx=0.001)
                errors[-1].append(term * der)

        return list(reversed(errors))

    def train(self, inputs: list[list[float]], expected_outputs: list[list[float]], epochs: int):
        for _ in range(epochs):
            for input, expected_output in zip(inputs, expected_outputs):
                output = self._calculate_outputs(input)
                errors = self._calculate_errors(input, expected_output)
                for layer, layer_errors, net in zip(self.layers, errors, output):
                    for neuron, error in zip(layer, layer_errors):
                        neuron.update_weights(net, error)
