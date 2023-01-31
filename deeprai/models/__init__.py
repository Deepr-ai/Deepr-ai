import deeprai.engine.build_model as builder
from deeprai.models.feed_forward.feed_forward import FeedForward
from deeprai.models.convolutional.convolutional import Convolutional
from deeprai.tools.file_manager.save import Save
from deeprai.tools.file_manager.load import Load

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

    def save(self, file_location):
        file = Save(file_location)
        file.save()

    def load(self, file_location):
        Load(file_location)

    def update_network(self):
        self.base_weights.update(self.weights)
        self.base_bias.update(self.bias)

    def clear_global_network(self):
        self.base_weights.pop()
        self.base_bias.pop()

    def clear_local_network(self):
        self.weights = []
        self.bias = []

