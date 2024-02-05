import cython
from deeprai.engine.cython.conv.conv_compute import convolve2d, conv_backprop
from deeprai.engine.cython.conv.pooling import max_pooling2d, average_pooling2d, avg_pool_backprop, max_pool_backprop
from deeprai.engine.cython.dense_operations import flatten, cnn_forward_propagate, cnn_dense_backprop
from deeprai.engine.cython.loss import mean_square_error, mean_absolute_error, categorical_cross_entropy
from deeprai.engine.cython.activation import softmax_derivative
import numpy as np
from deeprai.engine.base_layer import WeightVals
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
                                       str loss_type,
                                       list driv_list):
    cdef float epsilon = 1e-7
    cdef np.ndarray clipped_predictions, loss
    clipped_predictions = np.clip(predicted_output, epsilon, 1 - epsilon)
    # Loss calculations based on the loss type
    if loss_type == 'mean square error':
        loss = 2 * (predicted_output - true_output)
    elif loss_type == 'cross entropy':
        if driv_list[-1] == softmax_derivative:
            loss = predicted_output - true_output
        else:
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
                          list pool_size,
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
    delta = compute_initial_delta(final_output, true_output, loss_type, activation_derv_list)

    cdef int conv_pointer = len(conv_kernels) - 1
    cdef int dense_pointer = len(dense_weights) - 1
    cdef int pool_pointer = len(pool_size) - 1
    cdef int activation_pointer = len(activation_derv_list) -1

    for step in reversed(range(num_steps)):
        step_type, layer_input, layer_output, layer_input_shape, layer_output_shape = layer_outputs[step]
        if step_type == 'conv':
            layer_input = layer_outputs[conv_pointer][1]
            kernel = conv_kernels[conv_pointer]
            bias = conv_biases[conv_pointer] if use_bias else None
            activation_derv = activation_derv_list[conv_pointer]

            kernel_gradient, bias_gradient, new_delta = conv_backprop(delta, layer_input, kernel, bias, activation_derv,
                                                                      use_bias)
            conv_kernel_gradients.append(kernel_gradient)
            if use_bias:
                conv_bias_gradients.append(bias_gradient)
            delta = new_delta
            conv_pointer -= 1

        elif step_type == 'dense':
                layer_input = layer_outputs[step][1]
                weights = dense_weights[dense_pointer]
                biases = dense_biases[dense_pointer] if use_bias else None
                l1_pen = l1_penalty[dense_pointer]
                l2_pen = l2_penalty[dense_pointer]

                weight_gradient, bias_gradient, new_delta = cnn_dense_backprop(delta, layer_input, weights, biases,l1_pen,
                                                                           l2_pen, activation_derv_list[activation_pointer], use_bias)
                dense_weight_gradients.insert(0, weight_gradient)
                if use_bias:
                    dense_bias_gradients.insert(0, bias_gradient)
                delta = new_delta
                dense_pointer -= 1
                activation_pointer -=1

        elif step_type in ['avr_pool', 'max_pool']:
            layer_input = layer_outputs[step][1]
            pool_size_int = pool_size[pool_pointer]
            input_shape = layer_input_shape
            if delta.ndim == 1:
                output_shape = layer_output_shape
                delta_reshaped = delta.reshape(output_shape)
            else:
                delta_reshaped = delta
            if step_type == 'avg_pool':
                new_delta = avg_pool_backprop(delta_reshaped, pool_size_int, input_shape)
            elif step_type == 'max_pool':
                new_delta = max_pool_backprop(delta_reshaped, layer_input, pool_size_int, input_shape)
            delta = new_delta
            pool_pointer -= 1
    return conv_kernel_gradients, conv_bias_gradients, dense_weight_gradients, dense_bias_gradients


cpdef dict evaluate(np.ndarray inputs,
                    np.ndarray targets,
                    list operations, list operationStr, str loss_function_name):
    cdef int inputs_len = inputs.shape[0]
    cdef double sum_error = 0.0
    cdef np.ndarray abs_errors = np.zeros(inputs_len)
    cdef np.ndarray rel_errors = np.zeros(inputs_len)
    cdef np.ndarray[np.float64_t, ndim=1] output

    for i, (input_val, target) in enumerate(zip(inputs, targets)):
        output = cnn_forward_prop(input_val, operations, operationStr)

        if loss_function_name == "cross entropy":
            sum_error += categorical_cross_entropy(output, target)
        elif loss_function_name == "mean square error":
            sum_error += mean_square_error(output, target)
        elif loss_function_name == "mean absolute error":
            sum_error += mean_absolute_error(output, target)
        else:
            raise ValueError(f"Unsupported loss type: {loss_function_name}")

        abs_error = np.abs(output - target)
        rel_error = np.divide(abs_error, target, where=target != 0)
        abs_errors[i] = np.sum(abs_error)
        rel_errors[i] = np.mean(rel_error) * 100

    mean_rel_error = np.mean(rel_errors)
    total_rel_error = np.sum(rel_errors) / inputs_len
    accuracy = np.abs(100 - total_rel_error)

    return {
        "cost": sum_error / inputs_len,
        "accuracy": accuracy,
        "relative_error": total_rel_error
    }


