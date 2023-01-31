import math
import numpy as np
cimport numpy as np
#Public Functions for python wrapper

#none (input data)
cpdef np.ndarray[np.float64_t, ndim=1] linear(np.ndarray[np.float64_t, ndim=1] n):
    return n

cpdef np.ndarray[np.float64_t, ndim=1] linear_derivative(np.ndarray[np.float64_t, ndim=1] x):
    return np.ones_like(x)
#tanh
cpdef np.ndarray[np.float64_t, ndim=1] tanh(np.ndarray[np.float64_t, ndim=1] n):
    return np.tanh(n)

cpdef np.ndarray[np.float64_t, ndim=1] tanh_derivative(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] dy_dx = 1 - np.square(np.tanh(x))
    return dy_dx

#relu
cpdef np.ndarray[np.float64_t, ndim=1] relu(np.ndarray[np.float64_t, ndim=1] n):
    return np.maximum(n, 0)

cpdef np.ndarray[np.float64_t, ndim=1] relu_derivative(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] dy_dx = np.where(x > 0, 1, 0).astype(np.float64)
    return dy_dx

#leaky relu
cpdef np.ndarray[np.float64_t, ndim=1] leaky_relu(np.ndarray[np.float64_t, ndim=1] n, double alpha=0.01):
    if n < 0:
        return alpha * n
    else:
        return n
cpdef np.ndarray[np.float64_t, ndim=1] leaky_relu_derivative(np.ndarray[np.float64_t, ndim=2] x, alpha=0.01):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = np.where(x > 0, 1, alpha)
    return dy_dx

#Softmax
cpdef np.ndarray[np.float64_t, ndim=1] softmax(np.ndarray[np.float64_t, ndim=1] arr):
    cdef e = np.exp(arr)
    return e / e.sum()

cpdef np.ndarray[np.float64_t, ndim=1] softmax_derivative(np.ndarray[np.float64_t, ndim=2] y):
    cdef np.ndarray[np.float64_t, ndim=2] dy_dx = np.diagflat(y) - np.dot(y[:,np.newaxis], y[np.newaxis,:])
    return dy_dx

#Sigmoid
cpdef np.ndarray[np.float64_t, ndim=1] sigmoid(np.ndarray[np.float64_t, ndim=1] n):
    return 1.0 / (1.0 + np.exp(-n))

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

