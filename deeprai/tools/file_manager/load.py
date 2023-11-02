from pyntree import Node
from deeprai.engine.base_layer import LocalValues, LayerVals, BiasVals, WeightVals, NeuronVals, DerivativeVals, \
    BiasDerivativeVals
from deeprai.engine.build_model import Build
import numpy as np
class Load:
    def __init__(self, file_location):
        self.db = Node(file_location)
        builder = Build()

        # Loading network parameters
        WeightVals.Weights = self.db.Weight._val
        LayerVals.Layers = self.db.Layer._val
        NeuronVals.Neurons = self.db.Neuron._val
        BiasVals.Biases = self.db.Bias._val
        DerivativeVals.Derivatives = self.db.Derivative._val
        BiasDerivativeVals.BiasDerivatives = self.db.BiasDerivative._val

        # Loading local network settings
        LocalValues.DropoutList = self.db.DropoutList._val
        LocalValues.l1PenaltyList = self.db.l1PenaltyList._val
        LocalValues.l2PenaltyList = self.db.l2PenaltyList._val
        LocalValues.ActivationListString = self.db.ActivationListString._val
        LocalValues.OptimizerString = self.db.OptimizerString._val
        LocalValues.LossString = self.db.LossString._val


        for activation in LocalValues.ActivationListString[1:]:
            builder.convert_activations(activation)
            builder.convert_derivatives(activation)




    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if file_ext[-1] != "deepr":
            raise Exception(f"Can not import that file extension, use .deepr")
            return
