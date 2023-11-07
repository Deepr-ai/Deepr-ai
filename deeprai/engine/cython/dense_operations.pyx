import numpy as np
from deeprai.engine.cython import activation as act
import cython
from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals
cimport numpy as np
from libc.stdlib cimport rand
from deeprai.engine.cython.loss import categorical_cross_entropy, mean_square_error, mean_absolute_error
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs,
                                                         list activation_list, neurons, weights,
                                                         biases, bint use_bias, list dropout_rate,
                                                         bint training_mode=True):
    cdef int num_layers = len(neurons)
    cdef np.ndarray[np.float64_t, ndim=1] layer_outputs = inputs
    cdef int i, j
    cdef double mask_value

    # Clear previous neuron values
    NeuronVals.Neurons = []
    NeuronVals.Neurons.append(inputs)

    for i in range(num_layers - 1):
        if use_bias:
            layer_outputs = np.dot(layer_outputs, weights[i]) + biases[i]
        else:
            layer_outputs = np.dot(layer_outputs, weights[i])

        layer_outputs = activation_list[i](layer_outputs)

        # Store the output for each neuron
        NeuronVals.Neurons.append(layer_outputs)

        # Apply dropout regularization if in training mode
        if training_mode and dropout_rate[i] > 0:
            mask_value = 1 - dropout_rate[i]
            for j in range(layer_outputs.shape[0]):
                if np.random.binomial(1, mask_value) == 0:
                    layer_outputs[j] = 0.0

    return layer_outputs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple back_propagate(np.ndarray[np.float64_t, ndim=1] predicted_output,
                          np.ndarray[np.float64_t, ndim=1] true_output,
                          list activation_derv_list, neurons,
                          weights, list l1_penalty,
                          list l2_penalty, bint use_bias,
                          str loss_type='mean square error'):
    cdef int layer, num_layers = len(weights)
    cdef np.ndarray[np.float64_t, ndim=1] delta
    cdef np.ndarray[np.float64_t, ndim=2] weight_gradient
    cdef float l1, l2
    cdef float epsilon = 1e-7
    cdef np.ndarray clipped_predictions

    clipped_predictions = np.clip(predicted_output, epsilon, 1 - epsilon)
    # Loss calculations can be optimized based on the loss type
    if loss_type == 'mean square error':
        loss = 2 * (predicted_output - true_output)
    elif loss_type == 'cross entropy':
        loss = - (true_output / clipped_predictions) + (1 - true_output) / (1 - clipped_predictions)
    elif loss_type == 'mean absolute error':
        loss = np.sign(predicted_output - true_output)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    weight_gradients = []
    bias_gradients = []

    for layer in range(num_layers - 1, -1, -1):
        delta = loss * activation_derv_list[layer](neurons[layer + 1])
        weight_gradient = np.dot(neurons[layer].reshape(-1, 1), delta.reshape(1, -1))

        l1 = l1_penalty[layer]
        l2 = l2_penalty[layer]
        if l1 > 0:
            weight_gradient += l1 * np.sign(weights[layer])
        if l2 > 0:
            weight_gradient += 2 * l2 * weights[layer]

        weight_gradients.insert(0, weight_gradient)
        if use_bias:
            bias_gradients.insert(0, np.sum(delta, axis=0))

        loss = np.dot(delta, weights[layer].T)

    return weight_gradients, bias_gradients


cpdef dict evaluate(np.ndarray[np.float64_t, ndim=2] inputs,
                    np.ndarray[np.float64_t, ndim=2] targets,
                    list activation_list,
                    bint use_bias,
                    list dropout_rate,
                    str loss_function_name):
    cdef int inputs_len = inputs.shape[0]
    cdef double sum_error = 0.0
    cdef np.ndarray abs_errors = np.zeros(inputs_len)
    cdef np.ndarray rel_errors = np.zeros(inputs_len)
    cdef np.ndarray[np.float64_t, ndim=1] output

    for i, (input_val, target) in enumerate(zip(inputs, targets)):
        output = forward_propagate(input_val, activation_list, NeuronVals.Neurons, WeightVals.Weights, BiasVals.Biases, use_bias, dropout_rate)

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

# For CNN
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] flatten(np.ndarray[np.float64_t, ndim=2] input_2d):
    # Flatten the 2D input to a 1D array
    return input_2d.ravel()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple cnn_dense_backprop(np.ndarray[np.float64_t, ndim=1] delta,
                           np.ndarray[np.float64_t, ndim=1] layer_output,
                           np.ndarray[np.float64_t, ndim=2] weights,
                           object activation_derv,
                           bint use_bias):
    cdef int num_neurons = layer_output.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] activation_derv_values
    cdef np.ndarray[np.float64_t, ndim=2] weight_gradient
    cdef np.ndarray[np.float64_t, ndim=1] bias_gradient = np.empty(0)

    # Calculate derivative of activation function
    activation_derv_values = activation_derv(layer_output)

    # Calculate gradient for weights
    weight_gradient = np.dot(layer_output.reshape(-1, 1), delta.reshape(1, -1)) * activation_derv_values[:, None]
    if use_bias:
        bias_gradient = delta * activation_derv_values

    new_delta = np.dot(weights.T, delta) * activation_derv_values

    return new_delta, weight_gradient, bias_gradient
