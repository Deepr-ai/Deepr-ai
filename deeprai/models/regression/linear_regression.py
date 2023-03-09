from deeprai.engine.cython.regression import linear_regression


class LinearRegression:

    def __init__(self):
        self.fitted_vals = []

    def fit(self, x_vals, y_vals):
        out = linear_regression(x_vals=x_vals, y_vals=y_vals)
        self.fitted_vals = out
        return out

    def run(self, x_val):
        vals = self.fitted_vals
        return vals[0]*x_val+vals[1]
