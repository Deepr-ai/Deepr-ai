from deeprai.engine.cython import activation as act
class Relu:
    def relu(self, neuron):
        self.output = act.relu(neuron)
    def back_relu(self):
        pass



