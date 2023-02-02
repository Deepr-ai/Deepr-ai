from deeprai.engine.base_layer import WeightVals, LayerVals, KernelVals, ActivationList, ActivationDerivativeList, NeuronVals, \
    DerivativeVals, ActivationListString, ActivationDerivativeListString, Loss
import deeprai.engine.cython.activation as act
from deeprai.engine.cython import optimizers as opt
from deeprai.engine.cython import loss as lossFunc
import numpy as np
class Build:
    def __init__(self):
        self.NetworkQueue = []
        self.activationMap = {"tanh":act.tanh,"relu":act.relu,"leaky relu":act.leaky_relu,
                              "sigmoid":act.sigmoid, "linear":act.linear}
        self.activationDerivativeMap = {"tanh":act.tanh_derivative,"relu":act.relu_derivative,
                                        "leaky relu":act.leaky_relu_derivative,
                                        "sigmoid":act.sigmoid_derivative,"linear":act.linear_derivative}
        self.OptimizerMap = {"gradient decent": opt.gradient_descent}
        self.LossMap = {'mean square error': lossFunc.mean_square_error, "categorical cross entropy": lossFunc.categorical_cross_entropy,
                        "mean absolute error": lossFunc.mean_absolute_error}

    def create_kernel(self, amount, shape, max_size):
        self.NetworkQueue.append("kernel")
        local_kernels = []
        for val in range(amount):
            kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
            local_kernels.append(kernel.tolist())
        return local_kernels
    def create_pool(self):
        pass

    def convert_activations(self, activation):
        print(activation)
        ActivationList.append(lambda n: self.activationMap[activation](n))

    def convert_derivatives(self, activation): ActivationDerivativeList.append(lambda n: self.activationDerivativeMap[activation](n))

    def convert_loss(self, lossF): Loss[0]=(lambda o,t: self.LossMap[lossF[0]](o,t))

    def create_dense(self, size, activation='sigmoid'):
        #creats activation map
        ActivationListString.append(activation)
        ActivationDerivativeListString.append(activation)

        self.convert_activations(activation)
        self.convert_derivatives(activation)
        #works backwards to generate weight values
        layers = LayerVals.Layers
        NeuronVals.add(np.zeros(size))
        try:
            layers.append(size)
            WeightVals.add(np.random.rand(layers[-2], layers[-1]) * np.sqrt(2 / (layers[-2] + layers[-1])))
            DerivativeVals.add(np.zeros((layers[-2], layers[-1])))
        except IndexError:
            del ActivationList[0]
            del ActivationDerivativeList[0]
            #input neuron

    def create_flat(self):
        pass

    def create_merge(self):
        pass