import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] max_pooling2d(np.ndarray[np.float64_t, ndim=2] input,
                                                     int pool_size,
                                                     int stride):
    cdef int input_height = input.shape[0]
    cdef int input_width = input.shape[1]

    # Output dimensions
    cdef int output_height = (input_height - pool_size) // stride + 1
    cdef int output_width = (input_width - pool_size) // stride + 1

    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((output_height, output_width), dtype=np.float64)

    cdef int i, j, k, l
    cdef double max_val

    # Perform max pooling
    for i in range(output_height):
        for j in range(output_width):
            max_val = -np.inf
            for k in range(pool_size):
                for l in range(pool_size):
                    max_val = max(max_val, input[i * stride + k, j * stride + l])
            output[i, j] = max_val

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] average_pooling2d(np.ndarray[np.float64_t, ndim=2] input,
                                                         int pool_size,
                                                         int stride):
    cdef int input_height = input.shape[0]
    cdef int input_width = input.shape[1]

    # Output dimensions
    cdef int output_height = (input_height - pool_size) // stride + 1
    cdef int output_width = (input_width - pool_size) // stride + 1

    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((output_height, output_width), dtype=np.float64)

    cdef int i, j, k, l
    cdef double sum_val
    cdef int area = pool_size * pool_size

    # Perform average pooling
    for i in range(output_height):
        for j in range(output_width):
            sum_val = 0
            for k in range(pool_size):
                for l in range(pool_size):
                    sum_val += input[i * stride + k, j * stride + l]
            output[i, j] = sum_val / area

    return output

### BACK FUNCTIONS ###
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray average_pool_backprop(np.ndarray[np.float64_t, ndim=2] d_out,
                                       np.ndarray[np.float64_t, ndim=2] input,
                                       int pool_size,
                                       int stride):
    cdef int h, w, i, j, a, b
    cdef int H = input.shape[0], W = input.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] d_input = np.zeros_like(input)

    # Calculate the gradient for average pooling
    for h in range(d_out.shape[0]):
        for w in range(d_out.shape[1]):
            for i in range(pool_size):
                for j in range(pool_size):
                    a, b = h * stride + i, w * stride + j
                    if a < H and b < W:
                        d_input[a, b] += d_out[h, w] / (pool_size * pool_size)
    return d_input

cpdef np.ndarray max_pool_backprop(np.ndarray[np.float64_t, ndim=2] d_out,
                                   np.ndarray[np.float64_t, ndim=2] input,
                                   int pool_size,
                                   int stride,
                                   np.ndarray[np.int64_t, ndim=2] max_indices):
    cdef int h, w, i, j, a, b, max_h, max_w
    cdef int H = input.shape[0], W = input.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] d_input = np.zeros_like(input)

    # Calculate the gradient for max pooling
    for h in range(d_out.shape[0]):
        for w in range(d_out.shape[1]):
            max_h, max_w = max_indices[h, w]
            d_input[max_h, max_w] = d_out[h, w]
    return d_input