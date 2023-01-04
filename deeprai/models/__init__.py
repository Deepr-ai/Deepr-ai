import deeprai.engine.base_layer as base
import deeprai.engine.network_stack as stack


class Convolutional:
    def __init__(self):
        self.layers = []
        self.kernels = []
        self.bias = []
        self.padding = None
        self.strides = None
        self.output_labels = None
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

    def merge(self, merge_type='sum'):
        self.stack_layer.create_merge()

    def model_optimizers(self, optimizer='', loss=''):
        pass

    def train_model(self, input_data, label, verify_data=(), batch_size=10):
        pass

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

    def save(self, filename):
        pass

    def summery(self):
        pass


class FeedForward:
    def __init__(self):
        self.bias = []
        self.weights = []
        self.stack_layer = stack.StackEvents()
        self.base_weights = base.Weight(self.weights)
        self.base_bias = base.Bias(self.bias)


    def add_dense(self, neurons, activation=''):
        self.stack_layer.create_dense()

    def model_optimizers(self, optimizer='', loss=''):
        pass

    def train_model(self, input_data, label, verify_data=(), batch_size=10):
        pass

    def update_network(self):
        self.base_weights.update(self.weights)
        self.base_bias.update(self.bias)

    def clear_global_network(self):
        self.base_weights.pop()
        self.base_bias.pop()

    def clear_local_network(self):
        self.weights = []
        self.bias = []

    def save(self, filename):
        pass

    def summery(self):
        pass
