from deeprai.engine.base_layer import WeightVals, Optimizer, Loss, ActivationList, ActivationDerivativeList
import deeprai.engine.build_model as builder
from deeprai.engine.cython.dense_train_loop import train as train
from deeprai.engine.cython.dense_operations import forward_propagate
from deeprai.engine.cython import optimizers as opt
from deeprai.engine.cython import loss as lossFunc

class FeedForward:
    def __init__(self):
        self.spawn = builder.Build()
        self.OptimizerMap = {"gradient decent": opt.gradient_descent}
        self.LossMap = {'mean square error': lossFunc.mean_square_error, "categorical cross entropy": lossFunc.categorical_cross_entropy,
                        "mean absolute error": lossFunc.mean_absolute_error}


    def add_dense(self, neurons, activation=''):
        self.spawn.create_dense(neurons, activation)

    def model_optimizers(self, optimizer='gradient decent', loss='mean square error'):
        Optimizer[0] = optimizer
        Loss[0] = loss

    def train_model(self, input_data, verify_data, batch_size=10, epochs=500, learning_rate=0.1, momentum=0.6, verbose=True):
        loss_function = [lambda o,t: self.LossMap[Loss[0]](o,t)]
        print(ActivationList)
        train(inputs=input_data, targets=verify_data, epochs=epochs, learning_rate=learning_rate, momentum=momentum,
              activation_list=ActivationList, activation_derv_list=ActivationDerivativeList, loss_function=loss_function,
              verbose=verbose, batch_size=batch_size)

    def run(self, inputs):
        return forward_propagate(inputs, ActivationList)

    def save(self, filename):
        pass

    def summery(self):
        pass