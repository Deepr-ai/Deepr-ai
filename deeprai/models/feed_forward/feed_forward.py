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
            "adafactor": 0.001,
        }
        self.act_names = []

        self.__checkpoint_dir_loc = None
        self.__checkpoint_int = None

    def add_dense(self, neurons, activation='sigmoid', dropout=0, l1_penalty=0, l2_penalty=0):
        self.act_names.append(activation)
        self.spawn.create_dense(neurons, activation, dropout, l1_penalty, l2_penalty, self.use_bias)

    def config(self, optimizer='momentum', loss='mean square error', use_bias=True):
        self.opt_name = optimizer
        LossString[0] = loss
        self.use_bias = use_bias

    def train_model(self, train_inputs, train_targets, test_inputs, test_targets, batch_size=36, epochs=500,
                    learning_rate=None, momentum=0.6, verbose=True):
        learning_rate = self.learning_rate_dict[self.opt_name] if learning_rate == None else learning_rate
        self.__save_state()
        train(inputs=train_inputs, targets=train_targets, test_inputs=test_inputs, test_targets=test_targets,
              epochs=epochs, learning_rate=learning_rate, momentum=momentum,
              activation_list=ActivationList, activation_derv_list=ActivationDerivativeList, loss_function=LossString,
              verbose=verbose, batch_size=batch_size, dropout_rate=DropoutList, l1_penalty=l1PenaltyList,
              l2_penalty=l2PenaltyList, optimizer_name=self.opt_name, use_bias=self.use_bias, checkpoint_interval=self.__checkpoint_int,
              checkpoint_dir_location=self.__checkpoint_dir_loc)

    def run(self, inputs):
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

    def evaluate(self, inputs, targets, loss=None):
        # Support for custom loss not used in training
        if loss is None:
            loss = LossString[0]
        return evaluate(inputs, targets, ActivationList, self.use_bias, DropoutList, loss)

    def specs(self):
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
        print(f"\033[1mDeeprAI Version:\033[0m 1.0.3")
        print(divider)

    def graph(self, metric="cost-acc"):
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

    def save(self, file_location):
        file = Save(file_location)
        file.save()

    def load(self, file_location):
        Load(file_location)
        self.__load_state()

    def checkpoint(self, interval, dir_location):
        # Check if the directory exists; if not, create it
        if not os.path.exists(dir_location):
            os.makedirs(dir_location)
        self.__checkpoint_int = interval
        self.__checkpoint_dir_loc = dir_location

    def __save_state(self):
        # Saves a copy of each value to save to a file
        LocalValues.DropoutList = DropoutList
        LocalValues.LossString = LossString
        LocalValues.l2PenaltyList = l2PenaltyList
        LocalValues.l1PenaltyList = l1PenaltyList
        LocalValues.OptimizerString = self.opt_name
        LocalValues.ActivationListString = self.act_names

    def __load_state(self):
        # Loads values to use a model
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
    def report():
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
    adadelta = "adadelta"
    adafactor = "adafactor"
