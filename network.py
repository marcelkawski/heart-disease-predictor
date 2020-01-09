from layer import Layer
from neuron_types import Typee


class Network:

    def __init__(self, number_of_layers, reader, learning_rate, max_epochs, number_of_groups,
                 *numbers_of_neurons_in_layer):
        layers = []
        for counter, _ in enumerate(range(number_of_layers)):
            if counter != (number_of_layers - 1):
                if counter != 0:
                    # next layers' size (number of weights in a neuron) is the number of neurons in a previous layer
                    new_layer = Layer(Typee.SIGMOIDAL, numbers_of_neurons_in_layer[counter], len(layers[-1].neurons))
                else:
                    # first layer's size is the number of columns in each row of data
                    new_layer = Layer(Typee.SIGMOIDAL, numbers_of_neurons_in_layer[counter], reader.data_size)
            else:
                # The last layer must be linear
                new_layer = Layer(Typee.LINEAR, numbers_of_neurons_in_layer[counter], len(layers[-1].neurons))
            layers.append(new_layer)
        self.reader = reader
        self.layers = layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.number_of_groups = number_of_groups

    def forward_propagation(self, row):
        result = []
        i = 1
        #print("Layer: 0")
        output = self.layers[0].forward_propagation(row)
        while i < len(self.layers):
            #print("Layer: ", i)
            output = self.layers[i].forward_propagation(output)
            i += 1
        result.append(output)
        return result

    def backward_propagation(self, row):
        group = int(row[-1])
        for layer_counter, layer in reversed(list(enumerate(self.layers))):
            #print("Layer counter:", layer_counter)
            #print("Layer nr.: ", layer_counter)
            if layer.typee == Typee.LINEAR:
                for counter, neuron in enumerate(layer.neurons):
                    if group == counter:
                        expected = 1
                    else:
                        expected = 0
                    layer.neurons[counter].error = (expected - neuron.output) * neuron.derivative()
                    # print("!!!!!", group, counter, expected, neuron.output, neuron.derivative())
                    #print("Error: ", neuron.error)
            if layer.typee == Typee.SIGMOIDAL:
                for counter, neuron in enumerate(layer.neurons):
                    error = 0
                    for neuron2 in self.layers[layer_counter+1].neurons:
                        error += (neuron2.weights[counter] * neuron2.error)
                    layer.neurons[counter].error = error * neuron.derivative()
                    #print("Error: ", neuron.error)

    def update_weights(self, row):
        for layer_counter, layer in enumerate(self.layers):
            #print("Layer nr.: ", layer_counter)
            for neuron in layer.neurons:
                #print("\n")
                for counter, weight in enumerate(neuron.weights):
                    if layer_counter != 0:
                        # because there is one more weight (bias) than the number of weights in a neuron
                        # in the previous layer:
                        if counter < len(self.layers[layer_counter-1].neurons):
                            neuron.weights[counter] += self.learning_rate * neuron.error * \
                                                   self.layers[layer_counter-1].neurons[counter].output
                            #print("Weights: ", neuron.weights[counter])
                        else:
                            neuron.weights[counter] += self.learning_rate * neuron.error
                    else:
                        # because there is one more weights (bias) than columns in data:
                        if counter < len(row):
                            neuron.weights[counter] += self.learning_rate * neuron.error * float(row[counter])
                            #print("Weights: ", neuron.weights[counter])
                        else:
                            neuron.weights[counter] += self.learning_rate * neuron.error

    def train(self):
        for epoch in range(self.max_epochs):
            error = 0
            for row in self.reader.data:
                output = self.forward_propagation(row[:-1])
                output = [item for sublist in output for item in sublist]
                #print(output)
                expected = [0] * self.number_of_groups
                expected[int(row[-1])] = 1
                for i in range(self.number_of_groups):
                    error += ((expected[i] - output[i]) ** 2)
                self.backward_propagation(row)
                self.update_weights(row)
            print("Error: ", error)
            # print(self.layers[2].neurons[1].error)
