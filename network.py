from layer import Layer
from neuron_types import Typee
import math


class Network:

    def __init__(self, number_of_layers, reader, learning_rate, max_epochs, number_of_groups,
                 *numbers_of_neurons_in_layer):
        layers = []
        for counter, _ in enumerate(range(number_of_layers)):
            if counter != (number_of_layers - 1):
                if counter != 0:
                    # next layers' size (number of weights in a neuron) is the number of neurons in a previous layer
                    new_layer = Layer(Typee.SIGMOIDAL, numbers_of_neurons_in_layer[counter], len(layers[-1].neurons), max=int((1/math.sqrt(reader.data_size))*1000))
                else:
                    # first layer's size is the number of columns in each row of data
                    new_layer = Layer(Typee.SIGMOIDAL, numbers_of_neurons_in_layer[counter], reader.data_size, max=int((1/math.sqrt(reader.data_size))*1000))
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
        output = self.layers[0].forward_propagation(row)
        while i < len(self.layers):
            output = self.layers[i].forward_propagation(output)
            i += 1
        result.append(output)
        return result

    def backward_propagation(self, row):
        group = int(row[-1])
        for layer_counter, layer in reversed(list(enumerate(self.layers))):
            if layer.typee == Typee.LINEAR:
                for counter, neuron in enumerate(layer.neurons):
                    if group == counter:
                        expected = 1
                    else:
                        expected = 0
                    layer.neurons[counter].error = (expected - neuron.output) * neuron.derivative()
            if layer.typee == Typee.SIGMOIDAL:
                for counter, neuron in enumerate(layer.neurons):
                    error = 0
                    for neuron2 in self.layers[layer_counter+1].neurons:
                        error += (neuron2.weights[counter] * neuron2.error)
                    layer.neurons[counter].error = error * neuron.derivative()

    def update_weights(self, row):
        for layer_counter, layer in enumerate(self.layers):
            for neuron in layer.neurons:
                for counter, weight in enumerate(neuron.weights):
                    if layer_counter != 0:
                        # because there is one more weight (bias) than the number of weights in a neuron
                        # in the previous layer:
                        if counter < len(self.layers[layer_counter-1].neurons):
                            neuron.weights[counter] += self.learning_rate * neuron.error * \
                                                   self.layers[layer_counter-1].neurons[counter].output
                        else:
                            neuron.weights[counter] += self.learning_rate * neuron.error
                    else:
                        # because there is one more weights (bias) than columns in data:
                        if counter < len(row):
                            neuron.weights[counter] += self.learning_rate * neuron.error * float(row[counter])
                        else:
                            neuron.weights[counter] += self.learning_rate * neuron.error

    def train(self):
        for epoch in range(self.max_epochs):
            error = 0
            for row in self.reader.data:
                output = self.forward_propagation(row[:-1])
                output = [item for sublist in output for item in sublist]
                expected = [0] * self.number_of_groups
                expected[int(row[-1])] = 1
                for i in range(self.number_of_groups):
                    error += ((expected[i] - output[i]) ** 2)
                self.backward_propagation(row)
                self.update_weights(row)

    def test(self):
        correct_positives = 0
        correct_negatives = 0
        incorrect_positives = 0
        incorrect_negatives = 0

        for n, row in enumerate(self.reader.data):
            output = self.forward_propagation(row[:-1])[0]
            if output[0] > output[1]:
                predict = 0
            else:
                predict = 1

            real = int(row[-1])

            if predict == 1:
                if real == 1:
                    correct_positives += 1
                else:
                    incorrect_positives += 1
            else:
                if real == 1:
                    incorrect_negatives += 1
                else:
                    correct_negatives += 1

        print("CORRECT positive predictions: ", correct_positives)
        print("Incorrect positive predictions: ", incorrect_positives)
        print("CORRECT negative predictions: ", correct_negatives)
        print("Incorrect negative predictions: ", incorrect_negatives)

        correct_predictions = correct_positives + correct_negatives
        incorrect_predictions = incorrect_positives + incorrect_negatives

        print("\nEffectiveness of the network: ", correct_predictions / (correct_predictions+incorrect_predictions) * 100,
              " %")
