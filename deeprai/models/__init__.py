import deeprai.engine.build_model as builder
from deeprai.models.feed_forward.feed_forward import FeedForward
from deeprai.models.convolutional.convolutional import Convolutional
from deeprai.models.regression.linear_regression import LinearRegression
from deeprai.models.regression.sine_regression import SineRegression
from deeprai.models.regression.poly_regression import PolyRegression
from deeprai.models.KNN.KNN import KNN

class Convolutional(Convolutional):
    def __init__(self):
        self.spawn = builder.Build()
        super().__init__()

    def update_network(self):
        pass

    def clear_global_network(self):
        pass

    def clear_local_network(self):
        self.layers = []
        self.bias = []
        self.kernels = []


class FeedForward(FeedForward):
    def __init__(self):
        self.spawn = builder.Build()
        super().__init__()

    def update_network(self):
        self.base_weights.update(self.weights)
        self.base_bias.update(self.bias)

    def clear_global_network(self):
        self.base_weights.pop()
        self.base_bias.pop()

    def clear_local_network(self):
        self.weights = []
        self.bias = []


class KNN(KNN):
    def __int__(self):
        super().__init__()

class LinearRegression(LinearRegression):
    def __int__(self):
        super().__init__()


class PolyRegression(PolyRegression):
    def __int__(self):
        super().__init__()


class SineRegression(SineRegression):
    def __int__(self):
        super().__init__()