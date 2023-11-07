import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] convolve2d(np.ndarray[np.float64_t, ndim=2] input,
                                                  np.ndarray[np.float64_t, ndim=2] filter,
                                                  np.ndarray[np.float64_t, ndim=1] biases,
                                                  int stride, int padding,
                                                  object activation_func,
                                                  bint use_bias=False):
    cdef int filter_size = filter.shape[0]
    cdef int input_size = input.shape[0]

    # Apply padding to the input matrix
    cdef int padded_input_size = input_size + 2 * padding
    cdef np.ndarray[np.float64_t, ndim=2] padded_input = np.zeros((padded_input_size, padded_input_size),
                                                                  dtype=np.float64)
    padded_input[padding:-padding, padding:-padding] = input

    # Compute the dimensions of the output matrix
    cdef int output_size = ((input_size - filter_size + 2 * padding) // stride) + 1
    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((output_size, output_size), dtype=np.float64)

    cdef int i, j, k, l
    cdef double conv_sum

    # Perform the convolution with stride and padding
    for i in range(output_size):
        for j in range(output_size):
            conv_sum = 0
            for k in range(filter_size):
                for l in range(filter_size):
                    conv_sum += filter[k, l] * padded_input[i * stride + k, j * stride + l]

            # Add bias if the use_bias flag is True and biases are provided
            if use_bias and biases is not None:
                conv_sum += biases[0]

            # Apply the activation function
            output[i, j] = activation_func(conv_sum)

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double compute_single_convolution(np.ndarray[np.float64_t, ndim=2] d_out,
                                        np.ndarray[np.float64_t, ndim=2] filter,
                                        int i, int j, int padding):
    cdef int filter_height = filter.shape[0]
    cdef int filter_width = filter.shape[1]
    cdef double sum = 0
    cdef int m, n, input_x, input_y

    for m in range(filter_height):
        for n in range(filter_width):
            input_x = i - padding + m
            input_y = j - padding + n
            if 0 <= input_x < d_out.shape[0] and 0 <= input_y < d_out.shape[1]:
                sum += d_out[input_x, input_y] * filter[m, n]

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple conv_backprop(np.ndarray[np.float64_t, ndim=2] d_out,
                          np.ndarray[np.float64_t, ndim=2] input,
                          np.ndarray[np.float64_t, ndim=2] filter,
                          int stride,
                          int padding,
                          bint use_bias=False):
    cdef int filter_height = filter.shape[0]
    cdef int filter_width = filter.shape[1]
    cdef int d_out_height = d_out.shape[0]
    cdef int d_out_width = d_out.shape[1]
    cdef double d_bias = 0.0
    cdef int i, j

    # Prepare the output gradient arrays
    cdef np.ndarray[np.float64_t, ndim=2] d_input = np.zeros_like(input)
    cdef np.ndarray[np.float64_t, ndim=2] d_filter = np.zeros_like(filter)

    # Compute the gradient with respect to the filter and bias (if applicable)
    for i in range(d_out_height):
        for j in range(d_out_width):
            d_filter += input[i * stride:i * stride + filter_height,
                              j * stride:j * stride + filter_width] * d_out[i, j]
    if use_bias:
        d_bias = np.sum(d_out)

    # Compute the gradient with respect to the input
    flipped_filter = np.flip(filter)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            d_input[i, j] += compute_single_convolution(d_out, flipped_filter, i, j, padding)

    return (d_filter, d_input, d_bias) if use_bias else (d_filter, d_input)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double calculate_bias_gradient(np.ndarray[np.float64_t, ndim=2] d_out):
    cdef double bias_gradient = 0.0
    cdef int i, j

    # Sum over all positions in the feature map
    for i in range(d_out.shape[0]):
        for j in range(d_out.shape[1]):
            bias_gradient += d_out[i, j]

    return bias_gradient
