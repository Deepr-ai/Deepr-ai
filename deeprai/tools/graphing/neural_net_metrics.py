from deeprai.engine.base_layer import NetworkMetrics
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt

class MetricsGraphingEngine(object):

    def __init__(self):
        self.cost = NetworkMetrics[0]
        self.accuracy = NetworkMetrics[1]
        self.error = NetworkMetrics[2]
        self.epochs = NetworkMetrics[3]

    def graph_cost_and_accuracy(self):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost', color='tab:blue')
        ax1.plot(self.epochs, self.cost, color='tab:blue', label='Cost')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:orange')
        ax2.plot(self.epochs, self.accuracy, color='tab:orange', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.tight_layout()
        plt.title("Cost and Accuracy vs. Epoch")
        plt.show()

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
