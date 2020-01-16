import csv
from network import Network


class Reader:
    def __init__(self):
        self.data_size = 0
        self.data = []

    def get_data(self, data_file='heart.csv', encoding='utf-8', delimiter=','):
        data = []
        with open(data_file, encoding=encoding) as _file:
            reader = csv.reader(_file, delimiter=delimiter)
            for line in list(reader):
                data.append(line)
        data = data[1:]
        self.data_size = len(data)
        self.data = data


reader2 = Reader()
test_reader = Reader()

reader2.get_data(data_file="learn_data.csv")
network = Network(2, reader2, 0.3, 500, 2, 2, 2)
network.train()

test_reader.get_data(data_file="test_data.csv")
network.reader = test_reader
network.test()
