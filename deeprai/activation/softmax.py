from deeprai.engine.cython import activation as act
class Softmax:
    def softmax(self, neurons):
        self.output = act.softmax(neurons)

