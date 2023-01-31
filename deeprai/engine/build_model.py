from deeprai.engine.base_layer import WeightVals, LayerVals, KernelVals, ActivationList, ActivationDerivativeList, NeuronVals, DerivativeVals
import deeprai.engine.cython.activation as act
import numpy as np

class Build:
    def __init__(self):
        self.NetworkQueue = []
        self.activationMap = {"tanh":act.tanh,"relu":act.relu,"leaky relu":act.leaky_relu, "softmax":act.softmax,
                              "sigmoid":act.sigmoid}
        self.activationDerivativeMap = {"tanh":act.tanh_derivative,"relu":act.relu_derivative,
                                        "leaky relu":act.leaky_relu_derivative, "softmax":act.softmax_derivative,
                                        "sigmoid":act.sigmoid_derivative}

    def create_kernel(self, amount, shape, max_size):
        self.NetworkQueue.append("kernel")
        local_kernels = []
        for val in range(amount):
            kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
            local_kernels.append(kernel.tolist())
        return local_kernels
    def create_pool(self):
        pass
    def create_dense(self, size, activation='sigmoid'):
        #creats activation map
        ActivationList.append(lambda n: self.activationMap[activation](n))
        ActivationDerivativeList.append(lambda n: self.activationDerivativeMap[activation](n))
        #works backwards to generate weight values
        layers = LayerVals.Layers
        NeuronVals.add(np.zeros(size))
        try:
            layers.append(size)
            WeightVals.add(np.random.rand(layers[-2], layers[-1]) * np.sqrt(2 / (layers[-2] + layers[-1])))
            DerivativeVals.add(np.zeros((layers[-2], layers[-1])))
        except IndexError:
            pass
            #input neuron

    def create_flat(self):
        pass

    def create_merge(self):
        pass