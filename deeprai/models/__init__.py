import deeprai.engine.base_layer as base
import deeprai.engine.network_stack as stack


class Convolutional:
    def __init__(self, *input_matrix, padding=0, strides=0, output_labels=None):
        self.layers = [list(input_matrix)]
        self.kernels = []
        self.bias = []
        self.padding = padding
        self.strides = strides
        self.output_labels = output_labels
        self.stack_layer = stack.StackEvents()
        self.base_layer = base.Layer(self.layers)
        self.base_kernel = base.Kernel(self.kernels)
        self.base_bias = base.Bias(self.bias)

    def add_kernel(self, amount, shape, max_size: int = 2, ):
        self.stack_layer.create_kernel()

    def add_pool(self, shape):
        self.stack_layer.create_pool()

    def add_dense(self, neurons, activation=''):
        self.stack_layer.create_dense()

    def flatten(self):
        self.stack_layer.create_flat()

    def merge(self, type = 'sum'):
        self.stack_layer.create_merge()


    def update_network(self):
        self.base_kernel.update(self.layers)
        self.base_kernel.update(self.kernels)
        self.base_bias.update(self.bias)

    def clear_global_network(self):
        self.base_kernel.pop()
        self.base_layer.pop()
        self.base_layer.pop()

    def clear_local_network(self):
        self.layers = []
        self.bias = []
        self.kernels = []


class FeedForward:
    pass
