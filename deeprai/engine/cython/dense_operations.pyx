import numpy as np
from deeprai.engine.cython import activation as act
import cython
from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals
cimport numpy as np
from libc.stdlib cimport rand
from deeprai.engine.cython.loss import categorical_cross_entropy, mean_square_error, mean_absolute_error

@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs,
                                                         list activation_list, neurons, weights,
                                                         biases, bint use_bias, list dropout_rate,
                                                         bint training_mode=True):
    cdef int num_layers = len(neurons)
    cdef np.ndarray[np.float64_t, ndim=1] layer_outputs = inputs
    cdef int i

    # Clear previous neuron values
    NeuronVals.Neurons = []
    NeuronVals.Neurons.append(inputs)

    # Propagate through each layer in the network
    for i in range(num_layers - 1):
        # Calculate the weighted sum
        if use_bias:
            layer_outputs = np.dot(layer_outputs, weights[i]) + biases[i]
        else:
            layer_outputs = np.dot(layer_outputs, weights[i])

        # Apply the activation function
        layer_outputs = activation_list[i](layer_outputs)

        # Store the output for each neuron
        NeuronVals.Neurons.append(layer_outputs)

        # Apply dropout regularization if in training mode
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

    # Calculate the gradient of the loss function with respect to its inputs
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

    # Iterate backward through each layer to compute the gradient
    for layer in range(num_layers - 1, -1, -1):
        # Compute the delta (error term) for the layer
        delta = loss * activation_derv_list[layer](neurons[layer + 1])

        # Calculate the gradient for the weights using the outer product of the previous layer's output and delta
        weight_gradient = np.dot(neurons[layer].reshape(-1, 1), delta.reshape(1, -1))

        # Apply L1 and L2 regularization to the gradients
        l1 = l1_penalty[layer]
        l2 = l2_penalty[layer]
        if l1 > 0:
            weight_gradient += l1 * np.sign(weights[layer])
        if l2 > 0:
            weight_gradient += 2 * l2 * weights[layer]

        # Store the computed gradients
        weight_gradients.insert(0, weight_gradient)

        # If biases are used, compute and store their gradients
        if use_bias:
            bias_gradients.insert(0, np.sum(delta, axis=0))

        # Compute the gradient of the loss with respect to the output of the previous layer
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