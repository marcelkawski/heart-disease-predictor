from math import exp
from random import random
from neuron_types import _Type


class Neuron:

    def __init__(self, _type, data_size):
        self._type = _type
        weights = []
        for _ in range(data_size + 1):  # weights for each input and one for bias
            weight = random()
            weights.append(weight)
        self.weights = weights

    def adder(self, row):  # Assume that the vectors of data are the same length and there are no missing values.
        result = 0
        for counter, value in enumerate(row):
            result += float(value) * self.weights[counter]
        result += self.weights[-1]
        return result

    def activation_function(self, adder_result):
        if self._type == _Type.SIGMOIDAL:
            return 1.0 / (1.0 + exp(-adder_result))
        elif self._type == _Type.LINEAR:
            return adder_result

    def forward_propagation(self, row):
        return self.activation_function(self.adder(row))

