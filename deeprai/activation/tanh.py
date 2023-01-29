from deeprai.engine.cython import activation as act

class Tanh:
    def tanh(self, neuron):
        return act.tanh(neuron)

    def tanh_derivative(self, tanh):
        return act.tanh_derivative(tanh)
