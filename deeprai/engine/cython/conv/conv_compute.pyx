import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] convolve2d(np.ndarray input,
                                                  np.ndarray filter,
                                                  np.ndarray biases,
                                                  int stride, int padding,
                                                  activation_func,
                                                  bint use_bias=False):
    cdef int num_filters = filter.shape[0]
    cdef int filter_height = filter.shape[1]
    cdef int filter_width = filter.shape[2]
    cdef int input_depth = input.shape[0]
    cdef int input_height = input.shape[1]
    cdef int input_width = input.shape[2]

    # Apply padding to the input matrix
    cdef int padded_input_height = input_height + 2 * padding
    cdef int padded_input_width = input_width + 2 * padding
    cdef np.ndarray[np.float64_t, ndim=3] padded_input = np.zeros((input_depth, padded_input_height, padded_input_width), dtype=np.float64)

    # Handle padding
    if padding > 0:
        padded_input[:, padding:-padding, padding:-padding] = input
    else:
        padded_input = input

    # Compute the dimensions of the output matrix
    cdef int output_height = ((input_height - filter_height + 2 * padding) // stride) + 1
    cdef int output_width = ((input_width - filter_width + 2 * padding) // stride) + 1
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((num_filters, output_height, output_width), dtype=np.float64)

    cdef int n, i, j, k, l, m
    cdef double conv_sum

    # Perform the convolution for each filter
    for n in range(num_filters):
        for i in range(output_height):
            for j in range(output_width):
                conv_sum = 0.0
                for m in range(input_depth):  # Iterate over the depth
                    for k in range(filter_height):
                        for l in range(filter_width):
                            # Calculate the indices for the padded input
                            padded_i = i * stride + k
                            padded_j = j * stride + l
                            # Perform the convolution operation
                            conv_sum += filter[n, k, l] * padded_input[m, padded_i, padded_j]

                # Add bias if the use_bias flag is True and biases are provided
                if use_bias and biases is not None:
                    conv_sum += biases[n]

                # Apply the activation function
                # Fix later
                # output[n, i, j] = activation_func(conv_sum)

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple conv_backprop(np.ndarray[double, ndim=3] delta,
                          np.ndarray[double, ndim=3] layer_input,
                           conv_kernels,
                           conv_biases):
    conv_biases = np.array(conv_biases)
    conv_kernels = np.array(conv_kernels)
    # Get dimensions
    cdef int num_filters = conv_kernels.shape[0]
    cdef int filter_height = conv_kernels.shape[1]
    cdef int filter_width = conv_kernels.shape[2]
    cdef int num_channels = layer_input.shape[0]
    cdef int input_height = layer_input.shape[1]
    cdef int input_width = layer_input.shape[2]

    # Initialize gradients
    cdef np.ndarray d_kernels = np.zeros_like(conv_kernels)
    cdef np.ndarray d_biases = np.zeros_like(conv_biases)
    cdef np.ndarray prev_delta = np.zeros_like(layer_input)

    # Convolve delta with the flipped kernels to get prev_delta (correlation)
    cdef int i, j, k, m, n, f, y, x
    for f in range(num_filters):
        for y in range(filter_height):
            for x in range(filter_width):
                for m in range(input_height):
                    for n in range(input_width):
                        for i in range(num_channels):
                            if (m - y >= 0) and (n - x >= 0) and (m - y < delta.shape[1]) and (n - x < delta.shape[2]):
                                prev_delta[i, m, n] += delta[f, m - y, n - x] * conv_kernels[f, y, x]

    # Calculate gradients for weights (kernels)
    for f in range(num_filters):
        for y in range(filter_height):
            for x in range(filter_width):
                for m in range(delta.shape[1]):
                    for n in range(delta.shape[2]):
                        for i in range(num_channels):
                            if m + y < input_height and n + x < input_width:
                                d_kernels[f, y, x] += layer_input[i, m + y, n + x] * delta[f, m, n]

    # Calculate gradients for biases
    for f in range(num_filters):
        for m in range(delta.shape[1]):
            for n in range(delta.shape[2]):
                d_biases[f] += delta[f, m, n]

    return prev_delta, d_kernels, d_biases



