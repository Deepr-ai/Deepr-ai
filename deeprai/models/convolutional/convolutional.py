import deeprai.engine.build_model as builder
from deeprai.tools.graphing import neural_net_metrics
from deeprai.engine.base_layer import WeightVals, LocalValues, ActivationList, ActivationDerivativeList, LossString, \
    NeuronVals, DropoutList, l1PenaltyList, l2PenaltyList, LayerVals, BiasVals, NetworkMetrics, KernelVals, KernalBiasVals
from deeprai.engine.cython.conv.cnn_operations import cnn_forward_prop, evaluate
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
        self.conv_num = 0

    def add_pool(self, shape, type, stride=1):
        self.spawn.create_pool(shape, stride, type)

    def input_shape(self, size):
        self.input_size = size

    def add_conv(self, filters, kernel_size, stride=1, padding=0, actavation="relu"):
        if self.conv_num == 0:
            self._add_first_conv(filters, kernel_size, stride, padding, actavation)
            self.conv_num += 1
        else:
            kernels, biases = self.spawn.create_kernel(filters, kernel_size, filters)
            KernelVals.add(kernels)
            KernalBiasVals.add(biases)
            self.spawn.create_conv(stride, padding, self.use_bias, actavation)

    def _add_first_conv(self, filters, kernel_size, stride=1, padding=0, activation="relu"):
        _, _, input_depth = self.input_size
        kernels, biases = self.spawn.create_kernel(filters, kernel_size, input_depth)
        KernelVals.add(kernels)
        KernalBiasVals.add(biases)
        self.spawn.create_conv(stride, padding, self.use_bias, activation)

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
        return cnn_forward_prop(input, self.spawn.ConvList, self.spawn.ConvListStr)

    def train_model(self, train_inputs, train_targets, test_inputs, test_targets, batch_size=36, epochs=500,
                    learning_rate=None, momentum=0.6, verbose=True):
        learning_rate = self.learning_rate_dict[self.opt_name] if learning_rate == None else learning_rate
        train(inputs=train_inputs, targets=train_targets, epochs=epochs, test_inputs=test_inputs,
              test_targets=test_targets, learning_rate=learning_rate,
              loss_function=LossString[0], dropout_rate=DropoutList, activation_derv_list=ActivationDerivativeList,
              l2_penalty=l2PenaltyList, l1_penalty=l1PenaltyList, use_bias=self.use_bias, verbose=verbose,
              batch_size=batch_size, optimizer_name=self.opt_name, action_list=self.spawn.ConvList,
              action_list_str=self.spawn.ConvListStr, pool_size=self.spawn.PoolSize)

    def cnn_specs(self):
        divider = "\033[1m" + "─" * 50 + "\033[0m"
        dense_counter = 0
        loss_table = {
            "cross entropy": "Cross entropy",
            "mean square error": "MSE",
            "mean absolute error": "MAE"
        }
        print(divider)
        print("\033[1mConvolutional Neural Network Model:\033[0m")
        print(divider)

        # Initialize total parameters count
        total_parameters = 0

        # Iterate through each layer
        for layer_idx, (layer_type, layer_params) in enumerate(self.spawn.ConvList):
            layer_type = self.spawn.ConvListStr[layer_idx]
            print(f"  \033[1mLayer-{(layer_idx + 1)}:\033[0m")
            print(f"    \033[1mType\033[0m: {layer_type.capitalize()}")

            if layer_type == 'conv':
                # in_channels, out_channels = layer_params
                print(f"    \033[1mKernel Size\033[0m: {np.array(KernelVals.Kernels[layer_idx][0]).shape}")
                print(f"    \033[1mNum Filters\033[0m: {len(KernelVals.Kernels[layer_idx])}")
                print(f"    \033[1mStride\033[0m: {layer_params[2]}")
                print(f"    \033[1mPadding\033[0m: {layer_params[3]}")

                # Calculate parameters for Conv layer
                conv_params = sum(np.array(KernelVals.Kernels[layer_idx][0]).shape)
                total_parameters += conv_params

            elif layer_type == 'pool_max':
                pass

            elif layer_type == 'dense':
                print(f"    \033[1mIn_features\033[0m: {LayerVals.Layers[dense_counter]}")
                print(f"    \033[1mOut_features\033[0m: {LayerVals.Layers[dense_counter + 1]}")
                print(f"    \033[1mActivation\033[0m: {self.act_names[dense_counter].capitalize()}")
                print(f"    \033[1mDropout\033[0m: {DropoutList[dense_counter]:.2f}")
                print(f"    \033[1mL1 Penalty\033[0m: {l1PenaltyList[dense_counter]:.2f}")
                print(f"    \033[1mL2 Penalty\033[0m: {l2PenaltyList[dense_counter]:.2f}")

                fc_params = LayerVals.Layers[dense_counter] * LayerVals.Layers[dense_counter + 1]
                total_parameters += fc_params
                dense_counter +=1

            print(divider)

        print(f"\033[1mTotal Parameters:\033[0m {total_parameters}")
        print(f"\033[1mLoss Function:\033[0m {loss_table[LossString[0]]}")
        print(f"\033[1mOptimizer:\033[0m {self.opt_name.capitalize()}")
        print(f"\033[1mBias Usage:\033[0m {'Yes' if self.use_bias else 'No'}")
        print(f"\033[1mCNN Version:\033[0m 1.1.0")
        print(divider)

    def evaluate(self, inputs, targets, loss=None):
        # Support for custom loss not used in training
        if loss is None:
            loss = LossString[0]
        return evaluate(inputs, targets, self.spawn.ConvList, self.spawn.ConvListStr, loss)

    # Auto complete

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
    adafactor = "adafactor"
