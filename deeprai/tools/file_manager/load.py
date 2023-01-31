from pyndb import PYNDatabase
from deeprai.engine.base_layer import VelocityVals, WeightVals, NeuronVals, ActivationList, \
    ActivationDerivativeList, DerivativeVals, Optimizer, Loss
class Load:
    def __init__(self, file_location):
        self.format_file(file_location)
        self.db = PYNDatabase(file_location, filetype='pyndb')
        VelocityVals.Velocities = self.db.Velocity.val
        WeightVals.Weight = self.db.Weight.val
        NeuronVals.NeuronVals = self.db.NeuronVals.val
        ActivationList = self.db.Activation.val
        ActivationDerivativeList = self.db.ActivationDerivative.val
        DerivativeVals.Velocities = self.db.Velocity.val
        Optimizer = self.db.Optimizer.val
        Loss = self.db.Loss.val


    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if file_ext[-1] != "deepr":
            raise Exception(f"Can not import that file extension, use .deepr")
            return
