import cython
import numpy as np
cimport numpy as np
from scipy.ndimage import maximum_filter, zoom

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] max_pooling2d(np.ndarray input,
                                                     (int, int) pool_size,
                                                     int stride):
    cdef int input_depth = input.shape[0]
    cdef int input_height = input.shape[1]
    cdef int input_width = input.shape[2]
    cdef int pool_height = pool_size[0]
    cdef int pool_width = pool_size[1]

    # Output dimensions
    cdef int output_height = (input_height - pool_height) // stride + 1
    cdef int output_width = (input_width - pool_width) // stride + 1

    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((input_depth, output_height, output_width), dtype=np.float64)

    cdef int d, i, j, k, l
    cdef double max_val

    # Perform max pooling
    for d in range(input_depth):
        for i in range(output_height):
            for j in range(output_width):
                max_val = -np.inf
                for k in range(pool_height):
                    for l in range(pool_width):
                        max_val = max(max_val, input[d, i * stride + k, j * stride + l])
                output[d, i, j] = max_val

    return output


@cython.boundscheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] average_pooling2d(np.ndarray input,
                                                         (int, int) pool_size,
                                                         int stride):
    cdef int input_depth = input.shape[0]
    cdef int input_height = input.shape[1]
    cdef int input_width = input.shape[2]
    cdef int pool_height = pool_size[0]
    cdef int pool_width = pool_size[1]

    # Output dimensions
    cdef int output_height = (input_height - pool_height) // stride + 1
    cdef int output_width = (input_width - pool_width) // stride + 1

    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((input_depth, output_height, output_width), dtype=np.float64)

    cdef int d, i, j, k, l
    cdef double sum_val
    cdef int area = pool_height * pool_width

    # Perform average pooling
    for d in range(input_depth):
        for i in range(output_height):
            for j in range(output_width):
                sum_val = 0
                for k in range(pool_height):
                    for l in range(pool_width):
                        sum_val += input[d, i * stride + k, j * stride + l]
                output[d, i, j] = sum_val / area

    return output


### BACK FUNCTIONS ###
cpdef np.ndarray avg_pool_backprop(np.ndarray delta,
                                  int pool_size_int,
                                  tuple input_shape):
    cdef int num_filters = delta.shape[0] if delta.ndim == 3 else 1
    cdef np.ndarray new_delta

    # Convert pool_size to tuple
    pool_size = (pool_size_int, pool_size_int)

    if num_filters > 1:
        new_delta = np.zeros(input_shape, dtype=delta.dtype)
        for f in range(num_filters):
            scale_factors = (input_shape[1] / delta.shape[1], input_shape[2] / delta.shape[2])
            new_delta[f] = zoom(delta[f], scale_factors, order=1)
    else:
        scale_factors = (input_shape[0] / delta.shape[0], input_shape[1] / delta.shape[1])
        new_delta = zoom(delta, scale_factors, order=1)
    return new_delta

cpdef np.ndarray max_pool_backprop(np.ndarray delta,
                                  np.ndarray layer_input,
                                  int pool_size_int,
                                  tuple input_shape):
    cdef int num_filters = delta.shape[0] if delta.ndim == 3 else 1
    cdef np.ndarray new_delta

    pool_size = (pool_size_int, pool_size_int)

    if num_filters > 1:  # If delta is 3D
        new_delta = np.zeros(input_shape, dtype=delta.dtype)
        for f in range(num_filters):
            # Find the maxima locations for each filter
            max_locations = layer_input[f] == maximum_filter(layer_input[f], size=pool_size)
            scale_factors = (input_shape[1] / delta.shape[1], input_shape[2] / delta.shape[2])
            upscaled_delta = zoom(delta[f], scale_factors, order=1)
            new_delta[f] = upscaled_delta * max_locations
    else:  # If delta is 2D
        max_locations = layer_input == maximum_filter(layer_input, size=pool_size)
        scale_factors = (input_shape[0] / delta.shape[0], input_shape[1] / delta.shape[1])
        upscaled_delta = zoom(delta, scale_factors, order=1)
        new_delta = upscaled_delta * max_locations

    return new_delta