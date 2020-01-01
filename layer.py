from neuron import Neuron


class Layer:

    def __init__(self, _type, number_of_neurons, size_of_weights):
        self.typee = _type
        neurons = []
        for _ in range(number_of_neurons):
            new_neuron = Neuron(_type, size_of_weights)
            neurons.append(new_neuron)
        self.neurons = neurons

    def forward_propagation(self, _input):
        output_of_layer = []
        for neuron in self.neurons:
            print("Neuron", neuron)
            output_of_layer.append(neuron.forward_propagation(_input))
        return output_of_layer




