import numpy as np


def verify_inputs(array):
    return True if type(array) == np.ndarray else False


def round_out(array, a=2):
    np.set_printoptions(precision=a, suppress=True, floatmode='fixed')
    return np.round(array, a)
