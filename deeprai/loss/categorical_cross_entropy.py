from deeprai.engine.cython import loss as loss_function
class CategoricalCrossEntropy:
    def categorical_cross_entropy(self, outputs, targets):
        return loss_function.categorical_cross_entropy(outputs, targets)