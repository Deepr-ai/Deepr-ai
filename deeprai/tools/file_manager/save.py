from pyndb import PYNDatabase
from deeprai.engine.base_layer import VelocityVals, WeightVals, NeuronVals, ActivationList, \
    ActivationDerivativeList, DerivativeVals, Optimizer, Loss
class Save:
    def __init__(self, file_location):
        file_location = self.format_file(file_location)
        self.db = PYNDatabase(file_location, filetype='pyndb')
        self.FileLocation = file_location
        self.Velocity = VelocityVals.Velocities
        self.Weight = WeightVals.Weights
        self.NeuronVals = NeuronVals.Neurons
        self.Derivative = DerivativeVals.Derivatives
        self.Activation = ActivationList
        self.ActivationDerivative = ActivationDerivativeList
        self.Optimizer = Optimizer
        self.Loss = Loss

    def save(self):
        self.db.set("Velocity",self.Velocity)
        self.db.set("Weight", self.Weight)
        self.db.set("NeuronVals", self.NeuronVals)
        self.db.set("Derivative", self.Derivative)
        self.db.set("Activation", self.Activation)
        self.db.set("ActivationDerivative", self.ActivationDerivative)
        self.db.set("Optimizer", self.Optimizer)
        self.db.set("Loss", self.Loss)
        self.db.save(self.FileLocation)

    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if len(file_ext) == 1:
            raise Exception(f"Failed to specify file extension, try '{file_location}.deepr?' ")
            return
        file_ext[-1] = "deepr"
        return '.'.join(file_ext)

