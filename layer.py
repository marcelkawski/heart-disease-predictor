from neuron import Neuron


class Layer:

    def __init__(self, _type, number_of_neurons, size_of_weights, **kwargs):
        self.typee = _type
        self.neurons = [Neuron(_type, size_of_weights, max=kwargs.get("max")) if "max" in kwargs
                        else Neuron(_type, size_of_weights)
                        for _ in range(number_of_neurons)]

    def forward_propagation(self, _input):
        output_of_layer = []
        for neuron in self.neurons:

            output_of_layer.append(neuron.forward_propagation(_input))
        return output_of_layer




