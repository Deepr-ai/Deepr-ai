import cython
import numpy as np
cimport numpy as np

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
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=3] average_pool_backprop(np.ndarray delta,
                                                       np.ndarray[double, ndim=3] layer_output,
                                                       int pool_size,
                                                       int original_height,
                                                       int original_width):
    cdef int num_channels = layer_output.shape[0]
    cdef int h_out, w_out

    # Reshape delta depending on its dimensionality
    if delta.ndim == 1:
        h_out = original_height // pool_size
        w_out = original_width // pool_size
        delta = delta.reshape((num_channels, h_out, w_out))
    elif delta.ndim == 2:
        h_out = delta.shape[0]
        w_out = delta.shape[1]
        delta = delta.reshape((1, h_out, w_out))
    elif delta.ndim == 3:
        h_out = delta.shape[1]
        w_out = delta.shape[2]
    else:
        raise ValueError("Delta has an unexpected number of dimensions")

    cdef np.ndarray[double, ndim=3] prev_delta = np.zeros((num_channels, original_height, original_width), dtype=np.float64)
    cdef int ch, h, w, i, j
    cdef double grad = 1.0 / (pool_size * pool_size)

    # Propagate the gradients to the original dimensions
    for ch in range(num_channels):
        for h in range(h_out):
            for w in range(w_out):
                for i in range(pool_size):
                    for j in range(pool_size):
                        prev_delta[ch, h * pool_size + i, w * pool_size + j] = delta[ch, h, w] * grad

    return prev_delta


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] max_pool_backprop(np.ndarray[double, ndim=2] delta,
                                                   np.ndarray[double, ndim=2] layer_input,
                                                   np.ndarray[double, ndim=2] layer_output,
                                                   int pool_size):
    cdef int h_out = delta.shape[0]
    cdef int w_out = delta.shape[1]
    cdef int h, w, i, j, max_i, max_j
    cdef np.ndarray[double, ndim=2] prev_delta = np.zeros_like(layer_input)

    for h in range(h_out):
        for w in range(w_out):
            max_i, max_j = 0, 0
            # Find the position of the max value in the forward pass
            for i in range(pool_size):
                for j in range(pool_size):
                    if layer_output[h, w] == layer_input[h * pool_size + i, w * pool_size + j]:
                        max_i, max_j = h * pool_size + i, w * pool_size + j
                        break

            # Pass the gradient to the max position
            prev_delta[max_i, max_j] = delta[h, w]

    return prev_delta

