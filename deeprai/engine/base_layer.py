"""Keeps track of the network's values"""
class Layer:
    def __init__(self):
        # size of network for weight generation
        self.Layers = []

    def pop(self):
        self.Layers = []


class Bias:
    def __init__(self):
        self.Biases = []

    def pop(self):
        self.Biases = []

    def add(self, val):
        self.Biases.append(val)


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

class BiasDerivative:
    def __init__(self):
        self.BiasDerivatives = []

    def pop(self):
        self.BiasDerivatives = []

    def add(self, val):
        self.BiasDerivatives.append(val)

class LocalNetValues:
    def __init__(self):
        self.ActivationList = []
        self.DropoutList = []
        self.l1PenaltyList = []
        self.l2PenaltyList = []
        self.ActivationDerivativeList = []
        self.ActivationListString = []
        self.ActivationDerivativeListString = []
        self.OptimizerString = ['gradient decent']
        self.LossString = ['mean square error']


# GLOBAL NETWORK VALUES
WeightVals = Weight()
LayerVals = Layer()
NeuronVals = Neuron()
BiasVals = Bias()
DerivativeVals = Derivative()
BiasDerivativeVals = BiasDerivative()
LocalValues = LocalNetValues()
# LOCAL NETWORK VALUES
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
Optimizer = [[]]
DistanceIndex = [0]

# cost, acc, rel_error, epoch
NetworkMetrics=[[], [], [], []]
# Optimizer Values
