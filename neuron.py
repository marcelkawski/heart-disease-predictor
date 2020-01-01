from math import exp, atan
from random import random
from neuron_types import Typee


class Neuron:

    def __init__(self, typee, size_of_weights):
        self.typee = typee
        weights = []
        for _ in range(size_of_weights + 1):  # weights for each input and one for bias
            weight = random()
            weights.append(weight)
        self.weights = weights
        self.output = None
        self.error = None

    def adder(self, row):  # Assume that the vectors of data are the same length and there are no missing values.
        result = 0
        for counter, value in enumerate(row):
            result += float(value) * self.weights[counter]
        result += self.weights[-1]
        return result

    def activation_function(self, adder_result):
        if self.typee == Typee.SIGMOIDAL:
            return 1.0 / (1.0 + exp(-adder_result))
        elif self.typee == Typee.LINEAR:
            return adder_result
        else:
            print("ERROR: Need to implement an activation function for this type of neuron")
            exit(1)

    def forward_propagation(self, row):
        output = self.activation_function(self.adder(row))
        self.output = output
        #  print("Output", output)
        print(output)
        return output

    def derivative(self):
        if self.typee == Typee.SIGMOIDAL:
            return self.output * (1.0 - self.output)  # derivative of a sigmoidal function
        elif self.typee == Typee.LINEAR:
            return 1.0  # derivative for a linear function in output layer
        else:
            print("ERROR: Need to implement a derivative for this type of neuron")
            exit(1)

