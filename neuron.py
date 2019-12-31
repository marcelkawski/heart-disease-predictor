from math import atan
from types import Type


class Neuron:

    def __init__(self, _type):
        self.type = type

    @staticmethod
    def adder(_input):  # Assume that the vectors of data are the same length and there are no missing values.
        result = 0
        for value in _input:
            # TODO
            result += value  # * weight i
        # result += weight n+1
        return result

    def activation_function(self, result):
        if self.type == Type.SIGMOIDAL:
            result = atan(result)
            return result
        elif self.type == Type.LINEAR:
            return result
