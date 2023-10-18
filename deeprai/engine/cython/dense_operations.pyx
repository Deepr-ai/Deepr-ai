import numpy as np
from deeprai.engine.cython import activation as act
import cython
from deeprai.engine.base_layer import NeuronVals
cimport numpy as np
from libc.stdlib cimport rand

@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs,
                                                         list activation_list, neurons, weights,
                                                         biases, bint use_bias, list dropout_rate,
                                                         bint training_mode=True):
    cdef int num_layers = len(neurons)
    cdef np.ndarray[np.float64_t, ndim=1] layer_outputs = inputs
    cdef int i

    # Void previous neuron values
    NeuronVals.Neurons = []

    NeuronVals.Neurons.append(inputs)

    for i in range(num_layers - 1):
        if use_bias:
            layer_outputs = np.dot(layer_outputs, weights[i]) + biases[i]
        else:
            layer_outputs = np.dot(layer_outputs, weights[i])

        layer_outputs = activation_list[i](layer_outputs)
        NeuronVals.Neurons.append(layer_outputs)
        if training_mode and dropout_rate[i] > 0:
            mask = np.random.binomial(1, 1 - dropout_rate[i], size=layer_outputs.shape[0])
            layer_outputs *= mask

    return layer_outputs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple back_propagate(np.ndarray[np.float64_t, ndim=1] predicted_output,
                          np.ndarray[np.float64_t, ndim=1] true_output,
                          list activation_derv_list, neurons,
                          weights, list l1_penalty,
                          list l2_penalty, bint use_bias,
                          str loss_type='mean square error'):
    cdef int layer, num_layers = len(weights)
    cdef np.ndarray[np.float64_t, ndim=1] loss, delta
    cdef np.ndarray[np.float64_t, ndim=2] weight_gradient
    cdef float l1, l2
    # Calculating the initial gradient of the loss
    if loss_type == 'mean square error':
        loss = 2 * (predicted_output - true_output)
    elif loss_type == 'cross entropy':
        loss = - (true_output / predicted_output) + (1 - true_output) / (1 - predicted_output)
    elif loss_type == 'mean absolute error':
        loss = np.sign(predicted_output - true_output)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    weight_gradients = []
    bias_gradients = []

    for layer in range(num_layers - 1, -1, -1):
        delta = loss * activation_derv_list[layer](neurons[layer + 1])
        # Calculating the gradient for weights
        weight_gradient = np.dot(neurons[layer].reshape(-1, 1), delta.reshape(1, -1))

        l1 = l1_penalty[layer]
        l2 = l2_penalty[layer]
        if l1 > 0:
            weight_gradient += l1 * np.sign(weights[layer])
        if l2 > 0:
            weight_gradient += 2 * l2 * weights[layer]

        # Storing the computed gradients
        weight_gradients.insert(0, weight_gradient)

        if use_bias:
            bias_gradients.insert(0, np.sum(delta, axis=0))

        # Updating the loss term for the next layer in the backward pass
        loss = np.dot(delta, weights[layer].T)
    return weight_gradients, bias_gradients

