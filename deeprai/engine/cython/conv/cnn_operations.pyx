import cython
from deeprai.engine.cython.conv.conv_compute import convolve2d, conv_backprop, calculate_bias_gradient
from deeprai.engine.cython.conv.pooling import max_pooling2d, average_pooling2d, average_pool_backprop, max_pool_backprop
from deeprai.engine.cython.dense_operations import flatten, forward_propagate, cnn_dense_backprop
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray cnn_forward_prop(np.ndarray input, weights, biases, kernels, list operations, list operations_string):
    cdef np.ndarray layer_output = input
    for layer in range(len(operations)):
        op = operations[0]
        args = operations[1]
        if operations_string[layer] == "conv":
            layer_output = op(layer_output, *args)
        elif operations_string[layer_output] == "dense":
            pass
        elif operations_string[layer_output] == "avr_pool":
            pass
        elif operations_string[layer_output] == "max_pool":
            pass
        elif operations_string[layer_output] == "flat":
            pass
        else:
            raise Exception(f"Error in backend CNN Forward [incorrect action type '{operations_string[layer_output]}']: Report in a GitHub Issue.")

    return layer_output

cpdef np.ndarray compute_initial_delta(np.ndarray predicted_output,
                                       np.ndarray true_output,
                                       str loss_type):
    cdef float epsilon = 1e-7
    cdef np.ndarray clipped_predictions, loss

    clipped_predictions = np.clip(predicted_output, epsilon, 1 - epsilon)
    # Loss calculations based on the loss type
    if loss_type == 'mean square error':
        loss = 2 * (predicted_output - true_output)
    elif loss_type == 'cross entropy':
        loss = - (true_output / clipped_predictions) + (1 - true_output) / (1 - clipped_predictions)
    elif loss_type == 'mean absolute error':
        loss = np.sign(predicted_output - true_output)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple cnn_back_prop(np.ndarray[double, ndim=2] final_output,
                          np.ndarray[double, ndim=2] true_output,
                          list layer_outputs,
                          list steps,
                          list activation_derv_list,
                          list weights,
                          list l1_penalty,
                          list l2_penalty,
                          bint use_bias,
                          str loss_type='mean square error'):
    cdef int num_steps = len(steps)
    cdef list deltas = []
    cdef list weight_gradients = []
    cdef list bias_gradients = []
    cdef np.ndarray[double, ndim=2] delta, weight_gradient, bias_gradient

    # Calculate initial delta based on the loss function and the final output
    delta = compute_initial_delta(final_output, true_output, loss_type)

    # Iterate over the layers in reverse for backpropagation
    for i in range(num_steps - 1, -1, -1):
        step = steps[i]

        if step == "conv":
            delta, weight_gradient = conv_backprop(delta, layer_outputs[i], weights[i])
        elif step == "avr_pool":
            delta = average_pool_backprop(delta, layer_outputs[i])

        elif step == "max_pool":
            delta = max_pool_backprop(delta, layer_outputs[i])
        elif step == "dense":
            delta, weight_gradient, bias_gradient = cnn_dense_backprop(delta, layer_outputs[i], weights[i],
                                                                   activation_derv_list[i], use_bias)
            weight_gradients.insert(0, weight_gradient)
            if use_bias:
                bias_gradients.insert(0, bias_gradient)

        # Add gradients to the lists
        if step in ["conv", "dense"]:
            weight_gradients.insert(0, weight_gradient)
            if use_bias and step == "dense":
                bias_gradients.insert(0, bias_gradient)

        deltas.insert(0, delta)

    return weight_gradients, bias_gradients
