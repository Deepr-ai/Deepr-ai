from pyntree import Node
from deeprai.engine.base_layer import LocalValues, LayerVals, BiasVals, WeightVals, NeuronVals, DerivativeVals, BiasDerivativeVals
import random
import string
class Save:
    def __init__(self, file_location):
        file_location = self.format_file(file_location)
        self.db = Node(file_location)
        self.FileLocation = file_location

        # Network values
        self.Weight = WeightVals.Weights
        self.Layer = LayerVals.Layers
        self.Neuron = NeuronVals.Neurons
        self.Bias = BiasVals.Biases
        self.Derivative = DerivativeVals.Derivatives
        self.BiasDerivative = BiasDerivativeVals.BiasDerivatives

        # Local values
        self.DropoutList = LocalValues.DropoutList
        self.l1PenaltyList = LocalValues.l1PenaltyList
        self.l2PenaltyList = LocalValues.l2PenaltyList
        self.ActivationListString = LocalValues.ActivationListString
        self.OptimizerString = LocalValues.OptimizerString
        self.LossString = LocalValues.LossString

    def save(self):
        # Saving network parameters
        self.db.set("Weight", self.Weight)
        self.db.set("Layer", self.Layer)
        self.db.set("Neuron", self.Neuron)
        self.db.set("Bias", self.Bias)
        self.db.set("Derivative", self.Derivative)
        self.db.set("BiasDerivative", self.BiasDerivative)

        # Saving local network settings
        self.db.set("DropoutList", self.DropoutList)
        self.db.set("l1PenaltyList", self.l1PenaltyList)
        self.db.set("l2PenaltyList", self.l2PenaltyList)
        self.db.set("ActivationListString", self.ActivationListString)
        self.db.set("OptimizerString", self.OptimizerString)
        self.db.set("LossString", self.LossString)

        self.db.save(self.FileLocation)

    def format_file(self, file_location):
        file_ext = file_location.split(".")
        if len(file_ext) == 1:
            backup_name = "backup_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)) + ".deepr"
            print(f"Failed to specify file extension, using backup file name '{backup_name}'")
            return backup_name
        else:
            file_ext[-1] = "deepr"
            return '.'.join(file_ext)




