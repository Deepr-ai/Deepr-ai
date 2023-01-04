import math
import numpy as np

cpdef double tanh(double n):
    return math.tanh(n)

cpdef double step(double n):
    if n < 0:
        return 0
    else:
        return 1

cpdef double relu(double n):
    if n < 0:
        return 0
    else:
        return n

cpdef double leaky_relu(double n, double alpha=0.01):
    if n < 0:
        return alpha * n
    else:
        return n

cpdef softmax(list arr):
    cdef e = np.exp(arr)
    return e / e.sum()

cpdef double sigmoid(double n):
    return 1.0 / (1.0 + math.exp(-n))