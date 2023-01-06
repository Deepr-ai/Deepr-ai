"""Keeps track of the networks values"""


class Layer:
    def __init__(self, layers):
        self.Layers = layers

    def pop(self):
        self.Layers = []

    def update(self, layers):
        self.Layers = layers


class Kernel:
    def __init__(self):
        self.Kernels = []

    def pop(self):
        self.Kernels = []

    def update(self, kernels):
        self.Kernels = kernels



class Bias:
    def __init__(self, bias):
        self.Biases = bias

    def pop(self):
        self.Biases = []

    def update(self, bias):
        self.Biases = bias


class Weight:
    def __init__(self, weight):
        self.Weights = weight

    def pop(self):
        self.Weights = []

    def update(self, weight):
        self.Weights = weight
