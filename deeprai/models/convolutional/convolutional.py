import deeprai.engine.build_model as builder
from deeprai.tools.graphing import neural_net_metrics
from deeprai.engine.base_layer import LossString, KernelVals
from deeprai.engine.cython.conv.cnn_operations import cnn_forward_prop
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

        self.__checkpoint_dir_loc = None
        self.__checkpoint_int = 1

    def add_pool(self, shape, type):
        self.spawn.create_pool(shape, type)

    def add_conv(self, filters, kernal_size, stride=0, padding=0, actavation="relu"):
        KernelVals.append(self.spawn.create_kernel(filters, kernal_size))
        self.spawn.cre

    def add_dense(self, neurons, activation='sigmoid', dropout=0, l1_penalty=0, l2_penalty=0):
        self.act_names.append(activation)
        self.spawn.create_dense(neurons, activation, dropout, l1_penalty, l2_penalty, self.use_bias)

    def flatten(self):
        self.spawn.create_flat()

    def config(self, optimizer='momentum', loss='mean square error', use_bias=True):
        self.opt_name = optimizer
        LossString[0] = loss
        self.use_bias = use_bias

    def run(self, input):
        return cnn_forward_prop(input, self.spawn.ConvList)

    def train_model(self, input_data, verify_data, batch_size=10, epoches=500, learning_rate=0.1, momentum=0.6,
                    verbose=True):
        pass

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
