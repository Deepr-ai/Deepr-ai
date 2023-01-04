from deeprai.engine.cython import activation as act

class Sigmoid:
    def sigmoid(self, neuron):
        self.output = act.sigmoid(neuron)
