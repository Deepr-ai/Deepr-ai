from deeprai.engine.base_layer import WeightVals, LayerVals, ActivationList, ActivationDerivativeList, \
    NeuronVals, BiasVals, DerivativeVals, ActivationListString, ActivationDerivativeListString, Loss, DropoutList, \
    l1PenaltyList,l2PenaltyList, DistanceIndex, OptimizerString, Optimizer, BiasDerivativeVals, KernelVals, KernalBiasVals
import deeprai.engine.cython.activation as act
from deeprai.engine.cython import optimizers as opt
from deeprai.engine.cython import loss as lossFunc
from deeprai.engine.cython.dense_operations import *
import numpy as np


class Build:
    def __init__(self):
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

    def convert_activations(self, activation):
        ActivationList.append(self.activationMap[activation])

    def convert_derivatives(self, activation):
        ActivationDerivativeList.append(self.activationDerivativeMap[activation])

    def convert_loss(self, lossF):
        Loss[0] = (self.LossMap[lossF[0]])

    def convert_optimizer(self, OptF):
        Optimizer[0] = (self.OptimizerMap[OptF[0]])

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
