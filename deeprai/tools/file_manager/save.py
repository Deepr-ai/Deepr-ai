from pyntree import Node
from deeprai.engine.base_layer import LocalValues, LayerVals, BiasVals, WeightVals, NeuronVals, DerivativeVals, \
    BiasDerivativeVals
import random
import string
import warnings


class Save:  # TODO: Refactor to make Save.save() less weird
    def __init__(self, file_location):
        """
        Save backend for FeedForward.save() function
        Args:
            file_location: The target file to save the model to
        """
        self.file_location = file_location
        file_location = self.parse_filename()
        self.db = Node(file_location)

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
        """
        Save the loaded data to the file specified by the parent class
        """
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

        self.db.save(self.file_location)

    def parse_filename(self):
        """
        Checks the filename to see if it ends in .deepr and modifies it if it doesn't
        Returns: The filename ending in .deepr
        """
        file_ext = self.file_location.split(".")
        if len(file_ext) == 1:  # File extension forgotten
            backup_name = self.file_location[0] + ".deepr"
            warnings.warn(f"Failed to specify file extension, file saved as '{backup_name}'")
            return backup_name
        else:
            file_ext[-1] = "deepr"
            return '.'.join(file_ext)
