from deeprai.engine.base_layer import NetworkMetrics
import matplotlib.pyplot as plt
import numpy as np


class MetricsGraphingEngine(object):

    def __init__(self):
        self.cost = NetworkMetrics[0]
        self.accuracy = NetworkMetrics[1]
        self.error = NetworkMetrics[2]
        self.epochs = NetworkMetrics[3]

    def graph_cost(self):
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.plot(self.epochs, self.cost)
        plt.show()

    def graph_accuracy(self):
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(self.epochs, self.accuracy)
        plt.show()

    def graph_rel_error(self):
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.plot(self.epochs, self.error)
        plt.show()