from deeprai.engine.cython import activation as act

class Tanh:
    def tanh(self, neuron):
        self.output = act.tanh(neuron)

