from math import exp, atan
from random import random
from neuron_types import Typee
import random


class Neuron:

    def __init__(self, typee, size_of_weights, **kwargs):
        """
        if there is "max" param in kwargs set weights between -max and max
        else between 0 and 1

        used to initialize hidden neurons weights between +- 1/sqrt(dim(input))

        neurons which are not sigmoidal are linear - those are output neurons initialized with 0
        """
        self.typee = typee
        if "max" in kwargs:
            weights = [random.randint(-1*kwargs.get("max"), kwargs.get("max"))/1000
                       if self.typee == Typee.SIGMOIDAL
                       else 0
                       for _ in range(size_of_weights + 1)]
        else:
            weights = [random.random()
                       if self.typee == Typee.SIGMOIDAL
                       else 0
                       for _ in range(size_of_weights+1)]

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
        return output

    def derivative(self):
        if self.typee == Typee.SIGMOIDAL:
            return self.output * (1.0 - self.output)  # derivative of a sigmoidal function
        elif self.typee == Typee.LINEAR:
            return 1.0  # derivative for a linear function in output layer
        else:
            exit(1)

