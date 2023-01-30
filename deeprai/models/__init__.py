from deeprai.engine.base_layer import WeightVals, Optimizer, Loss, ActivationList, ActivationDerivativeList
import deeprai.engine.build_model as builder
from deeprai.engine.cython.dense_train_loop import train as train
from deeprai.engine.cython import optimizers as opt
from deeprai.engine.cython import loss as lossFunc


class Convolutional:
    def __init__(self):
        self.layers = []
        self.kernels = []
        self.bias = []
        self.padding = None
        self.strides = None
        self.output_labels = None

    def add_kernel(self, amount, shape, max_size: int = 2, ):
        self.stack_layer.create_kernel(amount, shape, max_size)

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

    def train_model(self, input_data, verify_data, batch_size=10, epoches=500, learning_rate=0.1, momentum=0.6, verbose=True):
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
        self.spawn = builder.Build()
        self.OptimizerMap = {"gradient decent": opt.gradient_descent}
        self.LossMap = {'mean square error': lossFunc.mean_square_error, "categorical cross entropy": lossFunc.categorical_cross_entropy}


    def add_dense(self, neurons, activation=''):
        self.spawn.create_dense(neurons, activation)

    def model_optimizers(self, optimizer='gradient decent', loss='mean square error'):
        Optimizer[0] = optimizer
        Loss[0] = loss

    def train_model(self, input_data, verify_data, batch_size=10, epochs=500, learning_rate=0.1, momentum=0.6, verbose=True):
        loss_function = [lambda o,t: self.LossMap[Loss[0]](o,t)]
        print(ActivationList)
        train(inputs=input_data, targets=verify_data, epochs=epochs, learning_rate=learning_rate, momentum=momentum,
              activation_list=ActivationList, activation_derv_list=ActivationDerivativeList, loss_function=loss_function,
              verbose=verbose, batch_size=batch_size)

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
    def run(self, inputs):
        return
