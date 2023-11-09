import cython
from deeprai.engine.cython.conv.conv_compute import convolve2d, conv_backprop
from deeprai.engine.cython.conv.pooling import max_pooling2d, average_pooling2d, average_pool_backprop, max_pool_backprop
from deeprai.engine.cython.dense_operations import flatten, cnn_forward_propagate, cnn_dense_backprop
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object cnn_forward_prop(np.ndarray input, list operations, list operationStr, bint return_intermediate=False):
    cdef list layer_outputs = []
    if input.ndim == 2:
        input = input[np.newaxis, :, :]

    cdef np.ndarray layer_input, layer_output
    layer_output = input
    cdef int i, dim
    for i, (operation, args) in enumerate(operations):
        layer_input = layer_output
        layer_output = operation(layer_input, *args)
        if return_intermediate:
            # Convert shapes to Python tuples before storing
            input_shape = tuple([int(layer_input.shape[dim]) for dim in range(layer_input.ndim)])
            output_shape = tuple([int(layer_output.shape[dim]) for dim in range(layer_output.ndim)])
            layer_outputs.append((operationStr[i], layer_input, layer_output, input_shape, output_shape))

    if return_intermediate:
        return layer_output, layer_outputs
    else:
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
cpdef tuple cnn_back_prop(np.ndarray[double, ndim=1] final_output,
                          np.ndarray[double, ndim=1] true_output,
                          list layer_outputs,
                          list steps,
                          list activation_derv_list,
                          list conv_kernels,
                          list dense_weights,
                          list conv_biases,
                          list dense_biases,
                          list l1_penalty,
                          list l2_penalty,
                          bint use_bias,
                          str loss_type='mean square error',
                          dict layer_shapes=None):
    cdef int num_steps = len(steps)
    cdef list deltas = []
    cdef list conv_kernel_gradients = []
    cdef list dense_weight_gradients = []
    cdef list conv_bias_gradients = []
    cdef list dense_bias_gradients = []
    cdef np.ndarray delta, weight_gradient, bias_gradient

    # Calculate initial delta based on the loss function and the final output
    delta = compute_initial_delta(final_output, true_output, loss_type)

    cdef int conv_pointer = len(conv_kernels) - 1
    cdef int dense_pointer = len(dense_weights) - 1

    # Iterate over the layers in reverse for backpropagation
    for i in range(num_steps - 1, -1, -1):
        step = steps[i]
        step_type, layer_input, layer_output ,layer_input_shape, layer_output_shape = layer_outputs[i]
        print(len(layer_outputs[i][1]))
        if step_type == "conv":
            delta, kernel_gradient, bias_grad = conv_backprop(delta, layer_output, conv_kernels[conv_pointer],
                                                              conv_biases[conv_pointer])
            conv_kernel_gradients.insert(0, kernel_gradient)
            if use_bias:
                conv_bias_gradients.insert(0, bias_grad)
            conv_pointer -= 1

        elif step_type == "avr_pool":
            # Time spent debugging cnn_dense_backprop: 20m
            delta = average_pool_backprop(delta, layer_output, 2, layer_output_shape[1]*2, layer_output_shape[2]*2)

        elif step_type == "max_pool":
            delta = max_pool_backprop(delta, layer_input, layer_output, 2)

        elif step_type == "dense":
            # Time spent debugging cnn_dense_backprop: 4h :crying:
            delta, weight_gradient, bias_grad = cnn_dense_backprop(delta, layer_input, dense_weights[dense_pointer],
                                                                   activation_derv_list[dense_pointer], use_bias)
            dense_weight_gradients.insert(0, weight_gradient)
            if use_bias:
                dense_bias_gradients.insert(0, bias_grad)
            dense_pointer -= 1

        deltas.insert(0, delta)

    # return gradents for convolutional kernels and dense weights separately
    return conv_kernel_gradients, conv_bias_gradients, dense_weight_gradients, dense_bias_gradients


