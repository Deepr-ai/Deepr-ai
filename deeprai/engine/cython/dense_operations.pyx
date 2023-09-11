import numpy as np
from deeprai.engine.cython import activation as act
import cython
#PUBLIC FUNCTIONprint(models.run(x))clear
cimport numpy as np
from libc.stdlib cimport rand

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

    Returns:
    -------
    np.ndarray
        The final output after forward propagation
    """
    neurons[0] = inputs
    # activation_list -> a list of lambda functions
    cdef np.ndarray[np.float64_t, ndim = 1] layer_outputs
    cdef int n_layers = len(weights)
    cdef int last_layer_index = n_layers - 1
    cdef float dropout
    cdef float one_minus_dropout
    cdef np.ndarray[np.float64_t, ndim = 1] neuron_layer
    cdef np.ndarray[np.float64_t, ndim = 2] weight_layer

    for layer in range(n_layers):
        weight_layer = weights[layer]
        layer_outputs = np.dot(neurons[layer], weight_layer)
        neurons[layer+1] = activation_list[layer](layer_outputs)
        if layer < last_layer_index:
            dropout = dropout_rate[layer]
            if dropout > 0:
                one_minus_dropout = 1.0 - dropout
                neurons[layer + 1] *= np.random.binomial([np.ones_like(neurons[layer + 1])], one_minus_dropout)[0] * (1.0 / one_minus_dropout)
    return neurons[-1]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] back_propagate(np.ndarray[np.float64_t, ndim=1] loss, list activation_derv_list,
                                                      list neurons, list weights, list derv, list l1_penalty,
                                                      list l2_penalty):
    cdef int layer, num_layers = len(derv)
    cdef np.ndarray[np.float64_t, ndim=1] delta
    cdef np.ndarray[np.float64_t, ndim=2] delta_reshape, current_reshaped
    cdef float l1, l2
    cdef np.ndarray[np.float64_t, ndim=2] weights_layer

    for layer in range(num_layers - 1, -1, -1):
        delta = loss * activation_derv_list[layer](neurons[layer + 1])

        # Reshape without new memory allocation
        delta_reshape = delta.reshape(delta.shape[0], -1).T
        current_reshaped = neurons[layer].reshape(neurons[layer].shape[0], -1)

        # In-place dot product to avoid new memory allocation
        np.dot(current_reshaped, delta_reshape, out=derv[layer])

        weights_layer = weights[layer]
        l1 = l1_penalty[layer]
        l2 = l2_penalty[layer]

        if l1 > 0:
            np.add(derv[layer], l1 * np.sign(weights_layer), out=derv[layer])
        if l2 > 0:
            np.add(derv[layer], 2 * l2 * weights_layer, out=derv[layer])

        loss = np.dot(delta, weights_layer.T)

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






