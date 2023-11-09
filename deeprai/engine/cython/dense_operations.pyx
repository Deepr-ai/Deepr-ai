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
cpdef np.ndarray[np.float64_t, ndim=1] flatten(np.ndarray input):
    # Check the number of dimensions of the input array
    if input.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D or 3D array.")
    return input.ravel()


cpdef tuple cnn_dense_backprop(np.ndarray[double, ndim=1] delta,
                               np.ndarray[double, ndim=1] layer_input,
                               np.ndarray[double, ndim=2] weights,
                               object activation_derv,
                               bint use_bias):
    cdef int num_neurons = weights.shape[0]
    cdef int num_inputs = weights.shape[1]

    # Check if the dimensions match (they don't)
    assert num_inputs == layer_input.size
    assert num_neurons == delta.size

    # Prepare the gradients using the shape variables
    cdef np.ndarray[double, ndim=1] delta_out = np.zeros(num_inputs, dtype=np.double)
    cdef np.ndarray[double, ndim=2] weight_gradient = np.zeros((num_neurons, num_inputs), dtype=np.double)
    cdef np.ndarray[double, ndim=1] bias_grad = np.zeros(num_neurons, dtype=np.double) if use_bias else None

    for i in range(num_neurons):
        for j in range(num_inputs):
            weight_gradient[i, j] = delta[i] * layer_input[j] * activation_derv(layer_input[j])
            delta_out[j] += delta[i] * weights[i, j]

    if use_bias:
        for i in range(num_neurons):
            bias_grad[i] = delta[i]

    for i in range(num_inputs):
        delta_out[i] *= activation_derv(layer_input[i])

    return delta_out, weight_gradient, bias_grad

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] cnn_forward_propagate(np.ndarray[np.float64_t, ndim=1] input_data, weights,
                                                                      biases, activation_func,  bint use_bias, list dropout_rate,
                                                                      bint training_mode=True):
    cdef np.ndarray[np.float64_t, ndim=1] layer_output = input_data
    cdef double mask_value

    layer_output = np.dot(layer_output, weights)
    if use_bias:
        layer_output += biases

    layer_output = activation_func(layer_output)

    # if training_mode and dropout_rate[i] > 0:
    #     mask_value = 1 - dropout_rate[i]
    #     dropout_mask = np.random.binomial(1, mask_value, size=layer_output.shape)
    #     layer_output *= dropout_mask

    return layer_output


