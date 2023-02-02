
from deeprai.engine.base_layer import WeightVals, Optimizer, Loss, ActivationList, ActivationDerivativeList, LossString, OptimizerString, NeuronVals
import deeprai.engine.build_model as builder
from deeprai.engine.cython.dense_train_loop import train as train
from deeprai.engine.cython.dense_operations import forward_propagate
import numpy as np


class FeedForward:
    def __init__(self):
        self.spawn = builder.Build()


    def add_dense(self, neurons, activation='sigmoid'):
        self.spawn.create_dense(neurons, activation)

    def model_optimizers(self, optimizer='gradient decent', loss='mean square error'):
        OptimizerString[0] = optimizer
        LossString[0] = loss

    def train_model(self, input_data, verify_data, batch_size=10, epochs=500, learning_rate=0.1, momentum=0.6, verbose=True):
        self.spawn.convert_loss(LossString)
        train(inputs=input_data, targets=verify_data, epochs=epochs, learning_rate=learning_rate, momentum=momentum,
              activation_list=ActivationList, activation_derv_list=ActivationDerivativeList, loss_function=Loss,
              verbose=verbose, batch_size=batch_size)

    def run(self, inputs):
        return forward_propagate(inputs, ActivationList)

    def save(self, filename):
        pass

    def summery(self):
        pass