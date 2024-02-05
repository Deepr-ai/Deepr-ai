import numpy as np
import csv


class CSVtoNumpy:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def convert(self):
        data = []
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                data.append([float(value) for value in row])
        return np.array(data)


class NumpyToCSV:
    def __init__(self, numpy_array, header=None):
        self.numpy_array = numpy_array
        self.header = header

    def write(self, csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            if self.header:
                writer.writerow(self.header)
            for row in self.numpy_array:
                writer.writerow(row)
