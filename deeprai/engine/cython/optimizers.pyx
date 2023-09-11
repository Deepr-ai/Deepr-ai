from deeprai.engine.base_layer import WeightVals, DerivativeVals
import numpy as np
cimport numpy as np
import cython
#Standerd gradient decent
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] gradient_descent(float learning_rate):
    cdef int num_layers = len(WeightVals.Weights)
    cdef list weights = WeightVals.Weights
    cdef list derivatives = DerivativeVals.Derivatives

    for layer in range(num_layers):
        weights[layer] += derivatives[layer] * learning_rate
