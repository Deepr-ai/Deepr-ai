from deeprai.engine.base_layer import WeightVals, LocalValues, ActivationList, ActivationDerivativeList, LossString, \
    NeuronVals, DropoutList, l1PenaltyList, l2PenaltyList, LayerVals, BiasVals, NetworkMetrics
import deeprai.engine.build_model as builder
from deeprai.engine.cython.dense_train_loop import train as train
from deeprai.engine.cython.dense_operations import forward_propagate, evaluate
from deeprai.tools.graphing import neural_net_metrics
from deeprai.tools.file_manager.save import Save
from deeprai.tools.file_manager.load import Load
import os
import numpy as np


class FeedForward:
    def __init__(self):
        """
        Creates a feed-forward neural network
        """
        self.spawn = builder.Build()
        self.graph_engine = neural_net_metrics.MetricsGraphingEngine()
        self.use_bias = True
        self.opt_name = "momentum"
        self.learning_rate_dict = {
            "gradient descent": 0.01,
            "momentum": 0.01,
            "rmsprop": 0.001,
            "adagrad": 0.01,
            "adam": 0.001,
            "adadelta": 1.0,
        }
        self.act_names = []

        self.__checkpoint_dir_loc = None
        self.__checkpoint_int = 1

    def add_dense(self, neurons: int, activation: str = 'sigmoid', dropout: float = 0, l1_penalty: float = 0,
                  l2_penalty: float = 0) -> None:
        """
        Add a dense layer to the network
        Args:
            neurons: The number of neurons in the layer
            activation: Activation function - choices include "sigmoid", "tanh", "relu", "leaky relu", "linear", "softmax"
            dropout: Probability from 0 to 1 that a neuron will be discarded
            l1_penalty: L1 normalization value (punishes weights that are high to prevent overfitting)
            l2_penalty: L2 normalization value (punishes weights that are high to prevent overfitting)
        """
        self.act_names.append(activation)
        self.spawn.create_dense(neurons, activation, dropout, l1_penalty, l2_penalty, self.use_bias)

    def config(self, optimizer: str = 'momentum', loss: str = 'mean square error', use_bias: bool = True) -> None:
        """
        Configure the network
        Args:
            optimizer: The function used to optimize the network. Choices include: "gradient descent", "momentum", "rmsprop", "adagrad", "adam", "adadelta"
            loss: The function used to calculate error in the network. Choices include: "cross entropy", "mean square error", "mean absolute error"
            use_bias: Whether the network will use biases
        """
        self.opt_name = optimizer
        LossString[0] = loss
        self.use_bias = use_bias

    def train_model(self, train_inputs: "np.array", train_targets: "np.array", test_inputs: "np.array",
                    test_targets: "np.array", batch_size: int = 36, epochs: int = 500,
                    learning_rate: float = None, momentum: float = 0.6, verbose: bool = True) -> None:
        """
        Train the network on the training data provided
        Args:
            train_inputs: A 2D numpy array of training inputs
            train_targets: A 2D numpy array of training target (expected) values
            test_inputs: A 2D numpy array of different inputs to test the network against
            test_targets: A 2D numpy array of different targets/expected values to use in testing the network
            batch_size: The number of inputs per batch (optimization function is applied per-batch)
            epochs: Number of times to cycle through the training data
            learning_rate: How quickly the network should learn from the data - lower is more precise
            momentum: The velocity multiplier for the optimization function
            verbose: Provides live stats on network performance
        """
        learning_rate = self.learning_rate_dict[self.opt_name] if learning_rate == None else learning_rate
        self.__save_state()
        train(inputs=train_inputs, targets=train_targets, test_inputs=test_inputs, test_targets=test_targets,
              epochs=epochs, learning_rate=learning_rate, momentum=momentum,
              activation_list=ActivationList, activation_derv_list=ActivationDerivativeList, loss_function=LossString,
              verbose=verbose, batch_size=batch_size, dropout_rate=DropoutList, l1_penalty=l1PenaltyList,
              l2_penalty=l2PenaltyList, optimizer_name=self.opt_name, use_bias=self.use_bias,
              checkpoint_interval=self.__checkpoint_int,
              checkpoint_dir_location=self.__checkpoint_dir_loc)

    def run(self, inputs: "np.array") -> "np.array":
        """
        Args:
            inputs: 1 or 2-dimensional array of inputs to run the model on

        Returns: A numpy array of network outputs
        """
        inputs = np.array(inputs)
        if len(inputs.shape) == 2:
            results = []
            for input_row in inputs:
                result = forward_propagate(input_row, ActivationList, NeuronVals.Neurons, WeightVals.Weights,
                                           BiasVals.Biases, self.use_bias, DropoutList, training_mode=False)
                results.append(result)
            return np.array(results)
        else:
            return forward_propagate(inputs, ActivationList, NeuronVals.Neurons, WeightVals.Weights,
                                     BiasVals.Biases, self.use_bias, DropoutList, training_mode=False)

    def evaluate(self, inputs: "np.array", targets: "np.array", loss=None) -> dict:
        """
        Runs the model on the input data and provides statistics on network performance
        Args:
            inputs: A 2D numpy array of inputs
            targets: A 2D numpy array of target (expected) values
            loss: The function used to calculate error. Choices include: "cross entropy", "mean square error", "mean absolute error". Can be different from the loss function used in training.

        Returns: cost, accuracy, relative error

        """
        # Support for custom loss not used in training
        if loss is None:
            loss = LossString[0]
        return evaluate(inputs, targets, ActivationList, self.use_bias, DropoutList, loss)

    def specs(self) -> None:
        """
        Prints out information about the network
        """
        loss_table = {
            "cross entropy": "Cross entropy",
            "mean square error": "MSE",
            "mean absolute error": "MAE"
        }

        parameters = sum([LayerVals.Layers[i] * LayerVals.Layers[i + 1] for i in range(len(LayerVals.Layers) - 1)])

        divider = "\033[1m" + "─" * 50 + "\033[0m"
        print(divider)
        print("\033[1mFeed Forward Model:\033[0m")
        print(divider)
        for i in range(len(LayerVals.Layers) - 1):
            print(f"  \033[1mLayer-{i + 1}:\033[0m")
            print(f"    \033[1mType\033[0m: Linear")
            print(f"    \033[1mIn_features\033[0m: {LayerVals.Layers[i]}")
            print(f"    \033[1mOut_features\033[0m: {LayerVals.Layers[i + 1]}")
            print(f"    \033[1mActivation\033[0m: {self.act_names[i].capitalize()}")
            print(f"    \033[1mDropout\033[0m: {DropoutList[i]:.2f}")
            print(f"    \033[1mL1 Penalty\033[0m: {l1PenaltyList[i]:.2f}")
            print(f"    \033[1mL2 Penalty\033[0m: {l2PenaltyList[i]:.2f}")
            print(divider)

        print(f"\033[1mTotal Parameters:\033[0m {parameters}")
        print(f"\033[1mLoss Function:\033[0m {loss_table[LossString[0]]}")
        print(f"\033[1mOptimizer:\033[0m {self.opt_name.capitalize()}")
        print(f"\033[1mBias Usage:\033[0m {'Yes' if self.use_bias else 'No'}")
        print(f"\033[1mDeeprAI Version:\033[0m 1.1.0")
        print(divider)

    def graph(self, metric : str ="cost-acc"):
        """
        Uses matplotlib to graph a function of the given metric
        Args:
            metric: Choices include "cost-acc", "cost", "acc", "error"
        """
        if metric == "cost-acc":
            self.graph_engine.graph_cost_and_accuracy()
        elif metric == "cost":
            self.graph_engine.graph_cost()
        elif metric == "acc" or metric == "accuracy":
            self.graph_engine.graph_accuracy()
        elif metric == "error" or metric == "relative error":
            self.graph_engine.graph_rel_error()
        else:
            print(f"Invalid metric: {metric}")

    def save(self, file_location: str):
        """
        Save the model to a .deepr file
        Args:
            file_location: The filename to save the model to, ending in .deepr
        """
        file = Save(file_location)
        file.save()

    def load(self, file_location: str):
        """
        Load a model from a .deepr file
        Args:
            file_location: The location of the model
        """
        Load(file_location)
        self.__load_state()

    def checkpoint(self, interval: int, dir_location: str):
        """
        Saves the model to a series of new files at the given interval
        Args:
            interval: The number of epochs to run before saving
            dir_location: The directory to place the .deepr files - will be created if nonexistant
        """
        # Check if the directory exists; if not, create it
        if not os.path.exists(dir_location):
            os.makedirs(dir_location)
        self.__checkpoint_int = interval
        self.__checkpoint_dir_loc = dir_location

    def __save_state(self):
        """
        Saves a copy of each value to save to a file
        """
        LocalValues.DropoutList = DropoutList
        LocalValues.LossString = LossString
        LocalValues.l2PenaltyList = l2PenaltyList
        LocalValues.l1PenaltyList = l1PenaltyList
        LocalValues.OptimizerString = self.opt_name
        LocalValues.ActivationListString = self.act_names

    def __load_state(self):
        """
        Loads values to use a model
        """
        DropoutList.clear()
        DropoutList.extend(LocalValues.DropoutList)
        LossString.clear()
        LossString.extend(LocalValues.LossString)
        l2PenaltyList.clear()
        l2PenaltyList.extend(LocalValues.l2PenaltyList)
        l1PenaltyList.clear()
        l1PenaltyList.extend(LocalValues.l1PenaltyList)
        self.opt_name = LocalValues.OptimizerString
        self.act_names = LocalValues.ActivationListString

    @staticmethod
    def report() -> dict:
        """
        Provides cost, accuracy, and error data for each epoch after training
        Returns: A dictionary containing cost, accuracy, and error as lists of floating point numbers
        """
        return {"cost": NetworkMetrics[0],
                "accuracy": NetworkMetrics[1],
                "error": NetworkMetrics[2]}

    # auto compleate

    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky relu"
    sigmoid = "sigmoid"
    linear = "linear"
    softmax = "softmax"

    # Loss functions
    gradient_descent = "gradient descent"
    mean_square_error = "mean square error"
    cross_entropy = "cross entropy"
    mean_absolute_error = "mean absolute error"

    # Optimizers
    gradient_descent = "gradient descent"
    momentum = "momentum"
    rmsprop = "rmsprop"
    adagrad = "adagrad"
    adam = "adam"
    adafactor = "adafactor"
