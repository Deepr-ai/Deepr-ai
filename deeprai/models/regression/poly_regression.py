from deeprai.engine.cython.regression import poly_regression


class PolyRegression:
    def __init__(self):
        self.fitted_vals = []

    def fit(self, x_vals, y_vals):
        out = poly_regression(x_vals=x_vals, y_vals=y_vals)
        self.fitted_vals = out
        return out

    def run(self, x_val):
        vals = self.fitted_vals
        output = 0
        for degree, coeff in enumerate(reversed(vals)):
            output += (coeff * x_val ** degree)
        return output
