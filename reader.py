import csv
from network import Network


class Reader:
    def __init__(self):
        self.data_size = 0
        self.data = []
        self.header = []

    def get_data(self, data_file='heart.csv', encoding='utf-8', delimiter=','):
        with open(data_file, encoding=encoding) as _file:
            reader = csv.reader(_file, delimiter=delimiter)
            data = [line for line in list(reader)]
        self.header = data[0]
        data = data[1:]
        self.data_size = len(data)
        self.data = data

    def divide_descrete(self, descrete={"cp": [0,1,2,3], "restecg": [0,1,2], "slope": [1,2,3], "thal": [1,2,3]}):
        """
        splits descrete input parameters like "chest pain type" into multiple boolean inputs
        """
        for n, row in enumerate(self.data):
            new_row = []
            for value, name in zip(row, self.header):
                if name in descrete:
                    for i in descrete[name]:
                        if int(value) == i:
                            new_row.append(1.0)
                        else:
                            new_row.append(0.0)
                else:
                    new_row.append(float(value))
            self.data[n] = new_row

    def normalize(self):
        """
        normalize all data parameters to values between 0 and 1
        """
        max = [x for x in self.data[0]]
        min = [x for x in self.data[0]]

        for row in self.data:
            for n, param in enumerate(row):
                if float(param) > float(max[n]):
                    max[n] = float(param)
                if float(param) < float(min[n]):
                    min[n] = float(param)

        for row in self.data:
            for n, param in enumerate(row[:-1]):
                if max[n] != min[n]:
                    row[n] = (param - min[n]) / (max[n] - min[n])

