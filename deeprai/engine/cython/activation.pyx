import math
import numpy as np
cimport numpy as np
import cython
cimport cython

#Public Functions for python wrapper

#none (input data)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] linear(np.ndarray[np.float64_t, ndim=1] n):
    return n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] linear_derivative(np.ndarray[np.float64_t, ndim=1] x):
    return np.ones_like(1)
#tanh
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] tanh(np.ndarray[np.float64_t, ndim=1] n):
    return np.tanh(n)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] tanh_derivative(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] dy_dx = 1 - np.square(np.tanh(x))
    return dy_dx

#relu
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] relu(np.ndarray[np.float64_t, ndim=1] n):
    return np.maximum(n, 0)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] relu_derivative(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] dy_dx = np.where(x > 0, 1, 0).astype(np.float64)
    return dy_dx

#leaky relu

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] leaky_relu(np.ndarray[np.float64_t, ndim=1] n, double alpha=0.01):
    return np.where(n >= 0, n, alpha * n)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] leaky_relu_derivative(np.ndarray[np.float64_t, ndim=1] x, alpha=0.01):
    cdef np.ndarray[np.float64_t, ndim=1] dy_dx = np.where(x > 0, 1, alpha)
    return dy_dx

#Softmax
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] softmax(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] exp_x = np.exp(x)
    cdef np.float64_t sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] softmax_derivative(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=2] jacobian = np.zeros((x.shape[0], x.shape[0]))
    cdef Py_ssize_t i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                jacobian[i, j] = x[i] * (1.0 - x[i])
            else:
                jacobian[i, j] = -x[i] * x[j]
    return jacobian.sum(axis=1)

#Sigmoid
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] sigmoid(np.ndarray[np.float64_t, ndim=1] n):
    return 1.0 / (1.0 + np.exp(-n))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] sigmoid_derivative(np.ndarray[np.float64_t, ndim=1] x):
    return x * (1 - x)

#Private Functions for other cython files (faster)

cdef np.ndarray[np.float64_t, ndim=1] cython_tanh(np.ndarray[np.float64_t, ndim=1] n):
    return np.tanh(n)

cdef np.ndarray[np.float64_t, ndim=1] cython_tanh_derivative(np.ndarray[np.float64_t, ndim=2] x):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = 1 - np.square(np.tanh(x))
    return dy_dx

#relu
cdef np.ndarray[np.float64_t, ndim=1] cython_relu(np.ndarray[np.float64_t, ndim=1] n):
    if n < 0:
        return 0
    else:
        return n
cdef np.ndarray[np.float64_t, ndim=1] cython_relu_derivative(np.ndarray[np.float64_t, ndim=2] x):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = np.where(x > 0, 1, 0)
    return dy_dx

#leaky relu
cdef np.ndarray[np.float64_t, ndim=1] cython_leaky_relu(np.ndarray[np.float64_t, ndim=1] n, double alpha=0.01):
    if n < 0:
        return alpha * n
    else:
        return n
cdef np.ndarray[np.float64_t, ndim=1] cython_leaky_relu_derivative(np.ndarray[np.float64_t, ndim=2] x, alpha=0.01):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = np.where(x > 0, 1, alpha)
    return dy_dx

#Softmax
cdef np.ndarray[np.float64_t, ndim=1] cython_softmax(np.ndarray[np.float64_t, ndim=1] arr):
    cdef e = np.exp(arr)
    return e / e.sum()

cdef np.ndarray[np.float64_t, ndim=1] cython_softmax_derivative(np.ndarray[np.float64_t, ndim=2] y):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = np.diagflat(y) - np.dot(y[:,np.newaxis], y[np.newaxis,:])
    return dy_dx

#Sigmoid
cdef np.ndarray[np.float64_t, ndim=1] cython_sigmoid(np.ndarray[np.float64_t, ndim=1] n):
    return 1.0 / (1.0 + np.exp(-n))
cdef np.ndarray[np.float64_t, ndim=1] cython_sigmoid_derivative(np.ndarray[np.float64_t, ndim=1] x):
    return x * (1 - x)

