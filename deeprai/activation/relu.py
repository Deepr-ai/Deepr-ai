from deeprai.engine.cython import activation as act
class Relu:
    def relu(self, neuron):
        return act.relu(neuron)

    def relu_derivative(self, relu):
        return act.relu_derivative(relu)


