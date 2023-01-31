from pyndb import PYNDatabase
from deeprai.engine.base_layer import VelocityVals, WeightVals, NeuronVals, ActivationList, \
    ActivationDerivativeList, DerivativeVals, Optimizer, Loss
from deeprai.engine.build_model import Build
class Load:
    def __init__(self, file_location):
        builder = Build()
        self.format_file(file_location)
        self.db = PYNDatabase(file_location, filetype='pickled')
        VelocityVals.Velocities = self.db.Velocity.val
        WeightVals.Weight = self.db.Weight.val
        NeuronVals.NeuronVals = self.db.NeuronVals.val
        DerivativeVals.Derivatives = self.db.Derivative.val
        # ActivationList = [builder.convert_activations(val) for val in self.db.Activation.val]
        # ActivationDerivativeList = [builder.convert_derivatives(val) for val in self.db.ActivationDerivative.val]
        # Loss = [builder.convert_loss(val) for val in self.db.Loss.val]
        for activation, derivative in zip(self.db.Activation.val, self.db.ActivationDerivative.val):
            builder.convert_activations(activation)
            builder.convert_derivatives(derivative)
        builder.convert_loss(self.db.Loss.val)
        print(NeuronVals.NeuronVals)
        print(ActivationList)
        print(ActivationDerivativeList)
        print(Loss)



    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if file_ext[-1] != "deepr":
            raise Exception(f"Can not import that file extension, use .deepr")
            return
