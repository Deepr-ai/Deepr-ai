from deeprai.engine.cython import activation as act
class LeakyRelu:
    def leaky_relu(self, neuron):
        self.output = act.leaky_relu(neuron)

