import numpy as np
cimport numpy as np


cpdef tuple gradient_descent_update(list weights, list biases, list weight_gradients, list bias_gradients,
                                    float learning_rate, bint use_bias):
    cdef int i
    cdef int num_layers = len(weights)

    for i in range(num_layers):
        weights[i] -= learning_rate * weight_gradients[i]
        if use_bias:
            biases[i] -= learning_rate * bias_gradients[i]

    return weights, biases


cpdef tuple momentum_update(list weights, list biases, list weight_gradients, list bias_gradients,
                            list weight_velocity, list bias_velocity, float learning_rate, float beta, bint use_bias):
    cdef int i
    cdef int num_layers = len(weights)

    for i in range(num_layers):
        weight_velocity[i] = beta * weight_velocity[i] + learning_rate * weight_gradients[i]
        weights[i] -= weight_velocity[i]

        if use_bias:
            bias_velocity[i] = beta * bias_velocity[i] + learning_rate * bias_gradients[i]
            biases[i] -= bias_velocity[i]

    return weights, biases, weight_velocity, bias_velocity


cpdef tuple adam_update(list weights, list biases, list weight_gradients, list bias_gradients,
                        list weight_m, list weight_v, list bias_m, list bias_v,
                        float learning_rate, float beta1=0.9, float beta2=0.999,
                        float epsilon=1e-7, int t=1, bint use_bias=True):
    cdef int i
    cdef int num_layers = len(weights)

    for i in range(num_layers):
        weight_m[i] = beta1 * weight_m[i] + (1 - beta1) * weight_gradients[i]
        weight_v[i] = beta2 * weight_v[i] + (1 - beta2) * (weight_gradients[i] ** 2)
        weight_m_corrected = weight_m[i] / (1 - beta1 ** t)
        weight_v_corrected = weight_v[i] / (1 - beta2 ** t)
        weights[i] -= learning_rate * weight_m_corrected / (np.sqrt(weight_v_corrected) + epsilon)

        if use_bias:
            bias_m[i] = beta1 * bias_m[i] + (1 - beta1) * bias_gradients[i]
            bias_v[i] = beta2 * bias_v[i] + (1 - beta2) * (bias_gradients[i] ** 2)
            bias_m_corrected = bias_m[i] / (1 - beta1 ** t)
            bias_v_corrected = bias_v[i] / (1 - beta2 ** t)
            biases[i] -= learning_rate * bias_m_corrected / (np.sqrt(bias_v_corrected) + epsilon)

    return weights, biases, weight_m, weight_v, bias_m, bias_v


cpdef tuple rmsprop_update(list weights, list biases, list weight_gradients, list bias_gradients,
                           list weight_v, list bias_v, float learning_rate, float beta=0.9, float epsilon=1e-7,
                           bint use_bias=True):
    cdef int i
    cdef int num_layers = len(weights)

    for i in range(num_layers):
        weight_v[i] = beta * weight_v[i] + (1 - beta) * (weight_gradients[i] ** 2)
        weights[i] -= learning_rate * weight_gradients[i] / (np.sqrt(weight_v[i]) + epsilon)

        if use_bias:
            bias_v[i] = beta * bias_v[i] + (1 - beta) * (bias_gradients[i] ** 2)
            biases[i] -= learning_rate * bias_gradients[i] / (np.sqrt(bias_v[i]) + epsilon)

    return weights, biases, weight_v, bias_v


cpdef tuple adagrad_update(list weights, list biases, list weight_gradients, list bias_gradients,
                           list weight_accumulated_grad, list bias_accumulated_grad, float learning_rate,
                           float epsilon=1e-7, bint use_bias=True):
    cdef int i
    cdef int num_layers = len(weights)

    for i in range(num_layers):
        weight_accumulated_grad[i] += weight_gradients[i] ** 2
        adjusted_grad = weight_gradients[i] / (np.sqrt(weight_accumulated_grad[i]) + epsilon)
        weights[i] -= learning_rate * adjusted_grad

        if use_bias:
            bias_accumulated_grad[i] += bias_gradients[i] ** 2
            adjusted_bias_grad = bias_gradients[i] / (np.sqrt(bias_accumulated_grad[i]) + epsilon)
            biases[i] -= learning_rate * adjusted_bias_grad

    return weights, biases, weight_accumulated_grad, bias_accumulated_grad


cpdef tuple adadelta_update(list weights, list biases, list weight_gradients, list bias_gradients,
                            list weight_accumulated_grad, list bias_accumulated_grad,
                            list weight_delta_accumulated, list bias_delta_accumulated,
                            float rho=0.95, float epsilon=1e-6, bint use_bias=True):
    cdef int i
    cdef int num_layers = len(weights)
    cdef np.ndarray[np.float64_t, ndim=2] weight_update
    cdef np.ndarray[np.float64_t, ndim=1] bias_update

    for i in range(num_layers):
        # Update accumulated gradient
        weight_accumulated_grad[i] = rho * weight_accumulated_grad[i] + (1 - rho) * (weight_gradients[i] ** 2)

        # Compute the model update
        weight_update = -(np.sqrt(weight_delta_accumulated[i] + epsilon) /
                          np.sqrt(weight_accumulated_grad[i] + epsilon)) * weight_gradients[i]
        weights[i] += weight_update

        # Update the accumulated updates
        weight_delta_accumulated[i] = rho * weight_delta_accumulated[i] + (1 - rho) * (weight_update ** 2)

        if use_bias:
            # Update accumulated gradient for bias
            bias_accumulated_grad[i] = rho * bias_accumulated_grad[i] + (1 - rho) * (bias_gradients[i] ** 2)

            # Compute the bias update
            bias_update = -(np.sqrt(bias_delta_accumulated[i] + epsilon) /
                            np.sqrt(bias_accumulated_grad[i] + epsilon)) * bias_gradients[i]
            biases[i] += bias_update

            # Update the accumulated updates for bias
            bias_delta_accumulated[i] = rho * bias_delta_accumulated[i] + (1 - rho) * (bias_update ** 2)

    return weights, biases, weight_accumulated_grad, bias_accumulated_grad, weight_delta_accumulated, bias_delta_accumulated


cpdef tuple adafactor_update(list weights, list biases, list weight_gradients, list bias_gradients,
                             list weight_v, list bias_v, float learning_rate,
                             float beta1=0.9, float epsilon=1e-6, bint use_bias=True):
    cdef int i
    cdef int num_layers = len(weights)
    cdef np.ndarray[np.float64_t, ndim=2] weight_update
    cdef np.ndarray[np.float64_t, ndim=1] bias_update


    for i in range(num_layers):
        # Update the moving average for squared gradients (v)
        weight_v[i] = beta1 * weight_v[i] + (1 - beta1) * (weight_gradients[i] ** 2)

        # Compute weight update based on the square root of the v
        weight_update = -learning_rate * weight_gradients[i] / (np.sqrt(weight_v[i]) + epsilon)
        weights[i] += weight_update

        if use_bias:
            # Update the moving average for squared bias gradients (v)
            bias_v[i] = beta1 * bias_v[i] + (1 - beta1) * (bias_gradients[i] ** 2)

            # Compute bias update
            bias_update = -learning_rate * bias_gradients[i] / (np.sqrt(bias_v[i]) + epsilon)
            biases[i] += bias_update

    return weights, biases, weight_v, bias_v

