from layer import Layer
from neuron_types import _Type


class Network:

    def __init__(self, number_of_layers, reader, *numbers_of_neurons_in_layer):
        layers = []
        for counter, _ in enumerate(range(number_of_layers)):
            if counter != (number_of_layers - 1):  # The last layer must be linear
                new_layer = Layer(_Type.SIGMOIDAL, numbers_of_neurons_in_layer[counter], reader.data_size)
            else:
                # Each layer for one group (ill, not ill)
                new_layer = Layer(_Type.LINEAR, numbers_of_neurons_in_layer[counter], reader.data_size)
            layers.append(new_layer)
        self.reader = reader
        self.layers = layers

    def forward_propagation(self):
        result = []
        for row in self.reader.data:
            i = 1
            output = self.layers[0].forward_propagation(row)
            while i != len(self.layers):
                output = self.layers[i].forward_propagation(output)
                i += 1
            result.append(output)
        return result
