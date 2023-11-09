import deeprai.engine.build_model as builder
from deeprai.tools.graphing import neural_net_metrics
from deeprai.engine.base_layer import WeightVals, LocalValues, ActivationList, ActivationDerivativeList, LossString, \
    NeuronVals, DropoutList, l1PenaltyList, l2PenaltyList, LayerVals, BiasVals, NetworkMetrics, KernelVals
from deeprai.engine.cython.conv.cnn_operations import cnn_forward_prop
from deeprai.engine.cython.conv.conv_train_loop import train
from deeprai.tools.file_manager.save import Save
from deeprai.tools.file_manager.load import Load
import os
import numpy as np


class Convolutional:
    def __init__(self):
        self.spawn = builder.Build()
        self.graph_engine = neural_net_metrics.MetricsGraphingEngine()
        self.use_bias = True
        self.opt_name = "momentum"
        self.learning_rate_dict = {
            "gradient descent": 0.01,
            "momentum": 0.01,
            "rmsprop": 0.001,
            "adagrad": 0.01,
            "adam": 0.001,
            "adadelta": 1.0,
            "adafactor": 0.001,
        }
        self.act_names = []
        self.input_size = ()
        self.__checkpoint_dir_loc = None
        self.__checkpoint_int = 1

    def add_pool(self, shape, type, stride=1):
        self.spawn.create_pool(shape, stride, type)

    def input_shape(self, size):
        self.input_size = size

    def add_conv(self, filters, kernel_size, stride=1, padding=0, actavation="relu"):
        KernelVals.add(self.spawn.create_kernel(filters, kernel_size))
        self.spawn.create_conv(stride, padding, self.use_bias, actavation)

    def add_dense(self, neurons, activation='sigmoid', dropout=0, l1_penalty=0, l2_penalty=0):
        # Check if the flattened output size has been calculated or needs to be calculated
        if self.spawn.flattened_output_size is None and self._needs_flattening():
            self.spawn.create_flattened(self.input_size)

        self.act_names.append(activation)
        self.spawn.create_cnn_dense(neurons, activation, dropout, l1_penalty, l2_penalty, self.use_bias)

    def _needs_flattening(self):
        return any(layer != 'dense' for layer in self.spawn.ConvListStr)

    def flatten(self):
        self.spawn.create_flat()

    def config(self, optimizer='momentum', loss='mean square error', use_bias=True):
        self.opt_name = optimizer
        LossString[0] = loss
        self.use_bias = use_bias

    def run(self, input):
        return cnn_forward_prop(input, self.spawn.ConvList)

    def train_model(self, train_inputs, train_targets, test_inputs, test_targets, batch_size=36, epochs=500,
                    learning_rate=None, momentum=0.6, verbose=True):
        learning_rate = self.learning_rate_dict[self.opt_name] if learning_rate == None else learning_rate
        train(inputs=train_inputs, targets=train_targets, epochs=epochs, test_inputs=test_inputs,
              test_targets=test_targets, learning_rate=learning_rate,
              loss_function=LossString[0], dropout_rate=DropoutList, activation_derv_list=ActivationDerivativeList,
              l2_penalty=l2PenaltyList, l1_penalty=l1PenaltyList, use_bias=self.use_bias, verbose=verbose,
              batch_size=batch_size, optimizer_name=self.opt_name, action_list=self.spawn.ConvList,
              action_list_str=self.spawn.ConvListStr)

    def summery(self):
        pass

        # auto compleate

    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky relu"
    sigmoid = "sigmoid"
    linear = "linear"
    softmax = "softmax"

    # Loss functions
    gradient_descent = "gradient descent"
    mean_square_error = "mean square error"
    cross_entropy = "cross entropy"
    mean_absolute_error = "mean absolute error"

    # Optimizers
    gradient_descent = "gradient descent"
    momentum = "momentum"
    rmsprop = "rmsprop"
    adagrad = "adagrad"
    adam = "adam"
    adadelta = "adadelta"
    adafactor = "adafactor"
