from deeprai.engine.base_layer import WeightVals, DerivativeVals
import numpy as np
cimport numpy as np

#Standerd gradient decent
cdef np.ndarray[np.float64_t, ndim=1] gradient_descent(float learning_rate):
    for layer in range(len(WeightVals.Weights)):
        WeightVals.Weights[layer] += DerivativeVals.Derivatives[layer] * learning_rate
