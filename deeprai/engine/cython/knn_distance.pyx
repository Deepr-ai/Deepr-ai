import numpy as np
cimport numpy as np
from libc.math cimport pow, sqrt

def euclidean_distance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    cdef int n = x.shape[0]
    cdef double dist = 0.0
    for i in range(n):
        dist += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(dist)

def manhattan_distance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    cdef int n = x.shape[0]
    cdef double dist = 0.0
    for i in range(n):
        dist += abs(x[i] - y[i])
    return dist

def minkowski_distance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y, int p=3):
    cdef int n = x.shape[0]
    cdef double dist = 0.0
    for i in range(n):
        dist += pow(abs(x[i] - y[i]), p)
    return pow(dist, 1.0/p)

def hamming_distance(np.ndarray[np.int_t, ndim=1] x, np.ndarray[np.int_t, ndim=1] y):
    cdef int n = x.shape[0]
    cdef int dist = 0
    for i in range(n):
        if x[i] != y[i]:
            dist += 1
    return dist
