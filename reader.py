import csv

DATA_FILE = 'heart.csv'


class Reader:

    @staticmethod
    def get_data():
        data = []
        with open(DATA_FILE, encoding='utf-8') as _file:
            reader = csv.reader(_file, delimiter=',')
            for line in list(reader):
                data.append(line)
        data = data[1:]
        return data
