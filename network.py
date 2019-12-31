from neuron import Neuron
from layer import Layer
from types import Type


class Network:

    def __init__(self, number_of_layers, *numbers_of_neurons_in_layer):
        layers = []
        for counter, _ in enumerate(number_of_layers):
            if counter != (number_of_layers - 1):  # The last layer must be linear
                new_layer = Layer(Type.SIGMOIDAL, numbers_of_neurons_in_layer[counter])
            else:
                new_layer = Layer(Type.LINEAR, 1)  # The result is only the 0/1 number
            layers.append(new_layer)
        self.layers = layers
