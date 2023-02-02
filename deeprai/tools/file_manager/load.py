from pyndb import PYNDatabase
from deeprai.engine.base_layer import VelocityVals, WeightVals, NeuronVals, ActivationList, \
    ActivationDerivativeList, DerivativeVals, Optimizer, LossString, LayerVals
from deeprai.engine.build_model import Build
import numpy as np
class Load:
    def __init__(self, file_location):
        builder = Build()
        self.format_file(file_location)
        self.db = PYNDatabase(file_location, filetype='pickled')
        VelocityVals.Velocities = self.db.Velocity.val
        WeightVals.Weights = self.db.Weight.val
        LayerVals.Layers = self.db.Layer.val
        NeuronVals.Neurons = self.db.Neuron.val
        DerivativeVals.Derivatives = self.db.Derivative.val
        LossString = self.db.Loss.val
        for activation, derivative in zip(self.db.Activation.val[1:], self.db.ActivationDerivative.val[1:]):
            builder.convert_activations(activation)
            builder.convert_derivatives(derivative)
        builder.convert_loss(self.db.Loss.val)




    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if file_ext[-1] != "deepr":
            raise Exception(f"Can not import that file extension, use .deepr")
            return
