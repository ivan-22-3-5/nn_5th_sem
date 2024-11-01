from src.neuron import Neuron


class Network:
    def __init__(self, *layer_sizes):
        self.layers = []
        for number_of_neurons, input_size in zip(layer_sizes[1:], layer_sizes):
            self.layers.append([Neuron(input_size) for _ in range(number_of_neurons)])

    def predict(self, inputs):
        for layer in self.layers:
            inputs = [neuron.predict(inputs) for neuron in layer]
        return inputs
