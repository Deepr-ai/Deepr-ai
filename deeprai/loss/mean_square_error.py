from deeprai.engine.cython import loss as loss_function
class MeanSquareError:
    def mean_square_error(self, outputs, targets):
        return loss_function.mean_square_error(outputs, targets)