from deeprai.engine.cython import loss as loss_function
class MeanAbsoluteError:
    def mean_absolute_error(self, outputs, targets):
        return loss_function.mean_absolute_error(outputs, targets)