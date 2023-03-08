from deeprai.engine.base_layer import WeightVals, DerivativeVals, MomentEstimateVals
import numpy as np
cimport numpy as np
import cython
#Standerd gradient decent
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] gradient_descent(float learning_rate):
    for layer in range(len(WeightVals.Weights)):
        WeightVals.Weights[layer] += DerivativeVals.Derivatives[layer] * learning_rate


# cpdef np.ndarray[np.float64_t, ndim=1] adam(int t, float learning_rate, float beta_1=0.9, float beta_2=0.999):
#     cdef int layer, rows, cols
#     moment_estimate_1 = MomentEstimateVals.moment_estimate_1
#     moment_estimate_2 = MomentEstimateVals.moment_estimate_2
#
#     # Loop through each layer
#     for layer in range(len(WeightVals.Weights)):
#         # Update the first and second moment estimates
#         moment_estimate_1[layer] = beta_1 * moment_estimate_1[layer] + (1 - beta_1) * DerivativeVals.Derivatives[layer]
#         moment_estimate_2[layer] = beta_2 * moment_estimate_2[layer] + (1 - beta_2) * np.square(DerivativeVals.Derivatives[layer])
#
#         # Calculate the weights update
#         weights_update = learning_rate * moment_estimate_1[layer] / (np.sqrt(moment_estimate_2[layer]) + 1e-8)
#
#         # Update the weights
#         WeightVals.Weights[layer] += weights_update
