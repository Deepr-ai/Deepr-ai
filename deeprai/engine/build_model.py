from deeprai.engine.base_layer import WeightVals, LayerVals, ActivationList, ActivationDerivativeList, \
    NeuronVals, BiasVals, DerivativeVals, ActivationListString, ActivationDerivativeListString, Loss, DropoutList, \
    l1PenaltyList,l2PenaltyList, DistanceIndex, OptimizerString, Optimizer, BiasDerivativeVals, KernelVals, KernalBiasVals
import deeprai.engine.cython.activation as act
from deeprai.engine.cython import optimizers as opt
from deeprai.engine.cython import loss as lossFunc
from deeprai.engine.cython.conv.conv_compute import *
from deeprai.engine.cython.conv.pooling import *
from deeprai.engine.cython.dense_operations import *
import numpy as np


class Build:
    def __init__(self):
        self.ConvList = []
        self.ConvListStr = []
        self.activationMap = {"tanh": act.tanh, "relu": act.relu, "leaky relu": act.leaky_relu,
                              "sigmoid": act.sigmoid, "linear": act.linear, "softmax": act.softmax}
        self.activationDerivativeMap = {"tanh": act.tanh_derivative, "relu": act.relu_derivative,
                                        "leaky relu": act.leaky_relu_derivative,
                                        "sigmoid": act.sigmoid_derivative, "linear": act.linear_derivative,
                                        "softmax": act.softmax_derivative}
        self.OptimizerMap = {
            "gradient_descent": opt.gradient_descent_update,
            "momentum": opt.momentum_update,
            "adam": opt.adam_update
        }
        self.LossMap = {'mean square error': lossFunc.mean_square_error,
                        "categorical cross entropy": lossFunc.categorical_cross_entropy,
                        "mean absolute error": lossFunc.mean_absolute_error}
        self.flattened_output_size = None

    def create_kernel(self, amount, shape, max_value=1):
        kernels = [np.random.randint(0, max_value, size=shape) for _ in range(amount)]
        KernalBiasVals.add(np.random.rand(amount).astype(np.float64))
        return kernels

    def create_pool(self, shape, stride, type):
        if type == "max":
            self.ConvList.append((max_pooling2d, (shape, stride)))
            self.ConvListStr.append("max_pool")
        elif type == "avr":
            self.ConvList.append((average_pooling2d, (shape, stride)))
            self.ConvListStr.append("avr_pool")

    def create_flat(self):
        self.ConvList.append((flatten, ()))
        self.ConvListStr.append("flat")

    def create_conv(self, stride, padding, use_bias, activation):
        self.ConvList.append((convolve2d, (
            np.array(KernelVals.Kernels[-1]), KernalBiasVals.KernalBias[-1], stride, padding, self.activationMap[activation], use_bias)))
        self.ConvListStr.append("conv")

    def create_flattened(self, input_shape):
        # Initialize current_shape with the assumption of a 3D input_shape
        # If input_shape is 2D, we convert it to 3D with a depth (channels) of 1
        current_shape = input_shape if len(input_shape) == 3 else (input_shape[0], input_shape[1], 1)

        if self.flattened_output_size is None:
            for i, layer in enumerate(self.ConvListStr):
                if layer == 'conv':
                    kernels, biases, stride, padding, activation_function, use_bias_flag = self.ConvList[i][1]
                    num_channels = len(kernels)
                    kernel_shape = kernels[0].shape

                    # Calculate output height and width
                    output_height = ((current_shape[0] - kernel_shape[0] + 2 * padding) // stride) + 1
                    output_width = ((current_shape[1] - kernel_shape[1] + 2 * padding) // stride) + 1

                    # Update current_shape for the next layer
                    current_shape = (output_height, output_width, num_channels)
                elif layer in ['max_pool', 'avr_pool']:
                    pool_shape, stride = self.ConvList[i][1]

                    # Calculate output height and width
                    output_height = ((current_shape[0] - pool_shape[0]) // stride) + 1
                    output_width = ((current_shape[1] - pool_shape[1]) // stride) + 1

                    # The number of output channels doesn't change after pooling
                    current_shape = (output_height, output_width, current_shape[2])
                elif layer == 'flat':
                    self.flattened_output_size = current_shape[0] * current_shape[1] * current_shape[2]
                    break

            if self.flattened_output_size is None:
                # If there are no layers that affect the size (no conv or pooling layers),
                # the flattened output size is just the product of the input dimensions.
                self.flattened_output_size = np.prod(input_shape)

        # Call create_dense with the calculated flattened output size
        self.create_dense(self.flattened_output_size)

    def create_cnn_dense(self, num_neurons, activation='sigmoid', dropout=0, l1_penalty=0, l2_penalty=0, use_bias=True):
        self.create_dense(num_neurons, activation, dropout, l1_penalty, l2_penalty, use_bias)
        # Note to self, the index of weights might mess up IDK
        self.ConvList.append((cnn_forward_propagate, (WeightVals.Weights[-1], BiasVals.Biases[-1],
                                                      self.activationMap[activation], use_bias, [dropout])))
        self.ConvListStr.append("dense")

    def convert_activations(self, activation):
        ActivationList.append(lambda n: self.activationMap[activation](n))

    def convert_derivatives(self, activation):
        ActivationDerivativeList.append(lambda n: self.activationDerivativeMap[activation](n))

    def convert_loss(self, lossF):
        Loss[0] = (lambda o, t: self.LossMap[lossF[0]](o, t))

    def convert_optimizer(self, OptF):
        Optimizer[0] = (lambda *args, **kwargs: self.OptimizerMap[OptF[0]](*args, **kwargs))

    def create_dense(self, size, activation='sigmoid', dropout=0, l1_penalty=0, l2_penalty=0, use_bias=True):
        # creates activation map
        ActivationListString.append(activation)
        ActivationDerivativeListString.append(activation)
        DropoutList.append(float(dropout))
        l1PenaltyList.append(float(l1_penalty))
        l2PenaltyList.append(float(l2_penalty))

        self.convert_activations(activation)
        self.convert_derivatives(activation)
        # works backwards to generate weight values
        layers = LayerVals.Layers
        NeuronVals.add(np.zeros(size))
        if use_bias:
            BiasVals.add(np.random.randn(size))  # Random biases for each neuron in the layer
            BiasDerivativeVals.add(np.zeros(size))

        try:
            layers.append(size)
            WeightVals.add(np.random.rand(layers[-2], layers[-1]) * np.sqrt(2 / (layers[-2] + layers[-1])))
            DerivativeVals.add(np.zeros((layers[-2], layers[-1])))
        except IndexError:
            del ActivationList[0]
            del ActivationDerivativeList[0]
            del DropoutList[0]
            del l2PenaltyList[0]
            del l1PenaltyList[0]
            if use_bias:
                del BiasVals.Biases[0]  # Remove bias for the input layer

    def translate_distance(self, distance):
        DistanceMap = {
            "euclidean distance": 0,
            "manhattan distance": 1,
            "minkowski distance": 2,
            "hamming distance": 3
        }
        # Directly modify the external DistanceIndex variable
        DistanceIndex[0] = DistanceMap[distance]
