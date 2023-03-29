import numpy as np
cimport numpy as np
from deeprai.engine.cython import activation as act
import cython
#PUBLIC FUNCTIONprint(models.run(x))clear

@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs, list activation_list, list neurons,
                                                         list weights, list dropout_rate):
    """
Parameters:
-----------
inputs : np.ndarray
    Input features to be propagated through the network
activation_list : list
    List of activation functions to be applied to the output of each layer
neurons : list
    List of np.ndarray objects to store the output of each layer
weights : list
    List of weights for each layer
l1_penalty : float
    L1 penalty term (default 0.0)
l2_penalty : float
    L2 penalty term (default 0.0)

Returns:
-------
np.ndarray
    The final output after forward propagation
"""
    neurons[0]= inputs
    # activation_list -> a list of lambda functions
    cdef np.ndarray[np.float64_t, ndim = 1] layer_outputs
    for layer, weight in enumerate(weights):
        layer_outputs = np.dot(neurons[layer], weight)
        neurons[layer+1] = activation_list[layer](layer_outputs)
        if layer < len(weights) - 1 and dropout_rate[layer] > 0:
            neurons[layer + 1] *= np.random.binomial([np.ones_like(neurons[layer + 1])], 1 - dropout_rate[layer])[0] * (
                        1.0 / (1 - dropout_rate[layer]))
    return neurons[-1]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] back_propagate(np.ndarray[np.float64_t, ndim=1] loss,  list activation_derv_list, list neurons, list weights, list derv, list l1_penalty,
                                                      list l2_penalty):
    """
Parameters:
-----------
loss : np.ndarray
    Loss to be backpropagated through the network
activation_derv_list : list
    List of derivative functions of the activation functions
neurons : list
    List of np.ndarray objects storing the output of each layer
weights : list
    List of weights for each layer
derv : list
    List of np.ndarray objects to store the gradient of the weights

Returns:
-------
np.ndarray
    The final gradient after backpropagation
"""
    cdef np.ndarray[np.float64_t, ndim = 1] delta
    cdef np.ndarray[np.float64_t, ndim = 2] delta_reshape, current_reshaped
    for layer in reversed(range(len(derv))):
        delta = loss * activation_derv_list[layer](neurons[layer+1])
        delta_reshape = delta.reshape(delta.shape[0], -1).T
        current_reshaped = neurons[layer].reshape(neurons[layer].shape[0], -1)
        derv[layer] = np.dot(current_reshaped, delta_reshape)
        if l1_penalty[layer] > 0:
            derv[layer] += l1_penalty[layer] * np.sign(weights[layer])
        if l2_penalty[layer] > 0:
            derv[layer] += 2 * l2_penalty[layer] * weights[layer]
        loss = np.dot(delta, weights[layer].T)

#PRIVATE FUNCTION
#note to self, add bies gradent
# cdef np.ndarray[np.float64_t, ndim=1] cython_back_propagate(np.ndarray[np.float64_t, ndim=1] loss,  list activation_derv_list ):
#     cdef np.ndarray[np.float64_t, ndim = 1] delta, delta_reshape, current_reshaped
#     for layer in reversed(range(len(derv))):
#         delta = loss * activation_derv_list[layer](neurons[layer+1])
#         delta_reshape = delta.reshape(delta.shape[0], -1).T
#         current_reshaped = neurons[layer].reshape(neurons[layer].shape[0], -1)
#         derv[layer] = np.dot(current_reshaped, delta_reshape)
#         loss = np.dot(delta, weights[layer].T)
#
#
# cdef np.ndarray[np.float64_t, ndim=1] cython_forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs, list activation_list):
#     neurons[0] = inputs
#     # activation_list -> a list of lambda functions
#     cdef np.ndarray[np.float64_t, ndim = 1] layer_outputs
#     for layer, weight in enumerate(weights):
#         layer_outputs = np.dot(neurons[layer], weight)
#         neurons[layer+1] = activation_list[layer](layer_outputs)
#     return neurons[-1]






