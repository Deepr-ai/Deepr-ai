import numpy as np
cimport numpy as np

#PUBLIC FUNCTIONS
cpdef float categorical_cross_entropy(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    outputs = np.clip(outputs, 1e-7, 1. - 1e-7)
    cdef float cross_entropy = -np.sum(targets * np.log(outputs + 1e-9)) / outputs.shape[0]
    return cross_entropy

cpdef float mean_square_error(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    return np.mean((targets - outputs)**2)

cpdef float mean_absolute_error(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    return np.mean(np.abs(outputs-targets))

#PRIVATE FUNCTIONS
cdef np.ndarray[np.float64_t, ndim=1] cython_categorical_cross_entropy(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    outputs = np.clip(outputs, 1e-7, 1. - 1e-7)
    cdef np.ndarray[np.float64_t, ndim=1] cross_entropy = -np.sum(targets * np.log(outputs + 1e-9)) / outputs.shape[0]
    return cross_entropy

cdef np.ndarray[np.float64_t, ndim=1] cython_mean_square_error(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    return np.mean((targets - outputs)**2)

cdef np.ndarray[np.float64_t, ndim=1] cython_mean_absolute_error(np.ndarray[np.float64_t, ndim=1] outputs, np.ndarray[np.float64_t, ndim=1] targets):
    return np.mean(np.abs(outputs-targets))