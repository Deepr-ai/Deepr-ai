import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list gradient_descent_update(list params, list grads, float learning_rate):
    cdef int i
    cdef int num_params = len(params)
    for i in range(num_params):
        params[i] -= learning_rate * grads[i]
    return params


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple momentum_update(list params, list grads, list velocity, float learning_rate, float beta=.09):
    cdef int i, j
    cdef int num_param_groups = len(params)

    for i in range(num_param_groups):
        for j in range(len(params[i])):
            velocity[i][j] = beta * velocity[i][j] + learning_rate * grads[i][j]
            params[i][j] -= velocity[i][j]
    return params, velocity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple adam_update(list params, list param_gradients, list param_m, list param_v,
                                   float learning_rate, float beta1=0.9, float beta2=0.999,
                                   float epsilon=1e-7, int t=1):
    cdef int i
    cdef int num_params = len(params)
    cdef float beta1_t = 1 - beta1 ** t
    cdef float beta2_t = 1 - beta2 ** t

    for i in range(num_params):
        param_m[i] = beta1 * param_m[i] + (1 - beta1) * param_gradients[i]
        param_v[i] = beta2 * param_v[i] + (1 - beta2) * np.square(param_gradients[i])
        params[i] -= learning_rate * (param_m[i] / beta1_t) / (np.sqrt(param_v[i] / beta2_t) + epsilon)

    return params, param_m, param_v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple rmsprop_update(list params, list grads, list velocity, float learning_rate, float beta=0.9, float epsilon=1e-7):
    cdef int i, j
    cdef int num_param_groups = len(params)

    for i in range(num_param_groups):
        for j in range(len(params[i])):
            velocity[i][j] = beta * velocity[i][j] + (1 - beta) * (grads[i][j] ** 2)
            params[i][j] = params[i][j].astype(np.float64)
            params[i][j] -= learning_rate * grads[i][j] / (np.sqrt(velocity[i][j]) + epsilon)

    return params, velocity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple adagrad_update(list params, list grads, list accumulated_grads, float learning_rate, float epsilon=1e-7):
    cdef int i, j
    cdef int num_param_groups = len(params)

    for i in range(num_param_groups):
        for j in range(len(params[i])):
            accumulated_grads[i][j] += grads[i][j] ** 2
            adjusted_grad = grads[i][j] / (np.sqrt(accumulated_grads[i][j]) + epsilon)
            params[i][j] = params[i][j].astype(np.float64)
            params[i][j] -= learning_rate * adjusted_grad

    return params, accumulated_grads


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef tuple adadelta_update(list params, list grads, list accumulated_grad, list delta_accumulated,
#                                        float rho=0.95, float epsilon=1e-6):
#     cdef int i
#     cdef np.ndarray update
#
#     for i in range(len(params)):
#         # Update accumulated gradient
#         accumulated_grad[i] = rho * accumulated_grad[i] + (1 - rho) * np.square(grads[i])
#
#         # Compute the model update
#         update = -(np.sqrt(delta_accumulated[i] + epsilon) /
#                    np.sqrt(accumulated_grad[i] + epsilon)) * grads[i]
#         params[i] += update
#
#         # Update the accumulated updates
#         delta_accumulated[i] = rho * delta_accumulated[i] + (1 - rho) * np.square(update)
#
#     return params, accumulated_grad, delta_accumulated


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple adafactor_update(list params, list grads, list second_moments, float learning_rate,
                                        float beta1=0.9, float epsilon=1e-6):
    cdef int i, j
    cdef int num_param_groups = len(params)
    cdef np.ndarray param_update

    for i in range(num_param_groups):
        for j in range(len(params[i])):
            # Update the moving average for squared gradients (v)
            second_moments[i][j] = beta1 * second_moments[i][j] + (1 - beta1) * (grads[i][j] ** 2)

            # Compute parameter update based on the square root of the second moment
            param_update = -learning_rate * grads[i][j] / (np.sqrt(second_moments[i][j]) + epsilon)
            params[i][j] = params[i][j].astype(np.float64)
            params[i][j] += param_update

    return params, second_moments


