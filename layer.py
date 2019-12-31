from types import Type
from neuron import Neuron


class Layer:

    def __init__(self, _type, number_of_neurons):
        self.type = _type
        neurons = []
        for _ in number_of_neurons:
            new_neuron = Neuron(_type)
            neurons.append(new_neuron)
        self.neurons = neurons

    def learn(self):
        pass  # TODO wyzaczanie optymalnych wag sieci neuronowej, np. za pomocą wstecznej propagacji gradientu
        # powinna zwracać listę wektorów wag (listę list) i dostarczac wagi neuronom
        # rozmiar o jeden wiekszy od rozmiaru danych (o 1 kolumne wiecej)
