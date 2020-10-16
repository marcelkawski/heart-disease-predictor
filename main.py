from reader import Reader
from network import Network

reader = Reader()
test_reader = Reader()

reader.get_data(data_file="data/learning_data.csv")
reader.divide_descrete()
reader.normalize()

network = Network(2, reader, 0.3, 500, 2, 8, 2)
network.train()

test_reader.get_data(data_file="data/testing_data.csv")
test_reader.divide_descrete()
test_reader.normalize()

network.reader = test_reader
network.test()
