from deeprai.engine.cython import activation as act
class Softmax:
    def softmax(self, neuron):
        return act.softmax(neuron)

    def softmax_derivative(self, softmax):
        return act.softmax_derivative(softmax)
