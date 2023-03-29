"""Keeps track of the network's values"""
import numpy as np


class Layer:
    def __init__(self):
        # size of network for weight generation
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
        self.Neurons = []

    def pop(self):
        self.Neurons = []

    def add(self, val):
        self.Neurons.append(val)


class Derivative:
    def __init__(self):
        self.Derivatives = []

    def pop(self):
        self.Derivatives = []

    def add(self, val):
        self.Derivatives.append(val)


# Spacific use cases (note to self, add condition statements)

class Velocity:
    def __init__(self):
        self.Velocities = None

    def pop(self):
        self.Velocities = []

class MomentEstimate:
    def __init__(self):
        self.moment_estimate_1 = []
        self.moment_estimate_2 = []

    def pop(self):
        self.Velocities = []


# GLOBAL NETWORK VALUES
WeightVals = Weight()
KernelVals = Kernel()
LayerVals = Layer()
NeuronVals = Neuron()
DerivativeVals = Derivative()
MomentEstimateVals = MomentEstimate()
# LOCAL NETWORK VALUES
VelocityVals = Velocity()
ActivationList = []
DropoutList = []
l1PenaltyList = []
l2PenaltyList = []
ActivationDerivativeList = []
ActivationListString = []
ActivationDerivativeListString = []
OptimizerString = ['gradient decent']
LossString = ['mean square error']
Loss = [[]]
Optimizer = []

# cost, acc, rel_error, epoch
NetworkMetrics=[[], [], [], []]
# Optimizer Values
