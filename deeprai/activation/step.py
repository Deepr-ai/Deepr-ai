from deeprai.engine.cython import activation as act
class Step:
    def step(self, neuron):
        self.output = act.step(neuron)

