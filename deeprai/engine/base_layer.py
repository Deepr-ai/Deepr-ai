"""Keeps track of the networks values"""
import numpy as np
class Layer:
    def __init__(self):
        #size of network for weight generation
        self.Layers = []

    def pop(self):
        self.Layers = []


class Kernel:
    def __init__(self):
        self.Kernels = np.array([])

    def pop(self):
        self.Kernels = np.array([])


class Bias:
    def __init__(self):
        self.Biases = np.array([])

    def pop(self):
        self.Biases = np.array([])



class Weight:
    def __init__(self):
        self.Weights = []

    def pop(self):
        self.Weights = []

    def add(self, val):
        self.Weights.append(val)


class Neuron:
    def __init__(self):
        self.Neurons = np.array([])

    def pop(self):
        self.Neurons = np.array([])

class Derivative:
    def __init__(self):
        self.Derivatives = np.array([])

    def pop(self):
        self.Derivatives = np.array([])

#Spacific use cases (note to self, add condition statements)

class Velocity:
    def __init__(self):
        self.Velocities = None
    def pop(self):
        self.Velocities = []

#GLOBAL NETWORK VALUES
WeightVals = Weight()
KernelVals = Kernel()
LayerVals = Layer()
NeuronVals = Neuron()
DerivativeVals = Derivative()
# LOCAL NETWORK VALUES
VelocityVals = Velocity()
ActivationList = []
ActivationDerivativeList = []
Optimizer = ['gradient decent']
Loss = ['mean square error']

