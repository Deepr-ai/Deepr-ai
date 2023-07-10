from deeprai.engine.cython.regression import sine_regression
from numpy import sin, pi


class SineRegression:
    def __init__(self):
        self.fitted_vals = []

    def fit(self, x_vals, y_vals):
        out = sine_regression(x_vals=x_vals, y_vals=y_vals)
        self.fitted_vals = out
        return out

    def run(self, x_val):
        vals = self.fitted_vals
        y = vals[0] * sin(2 * pi * vals[1] * (x_val - vals[2])) + vals[3]
        return y