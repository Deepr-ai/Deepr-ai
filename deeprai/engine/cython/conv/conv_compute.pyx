import cython
import numpy as np
cimport numpy as np
from scipy.signal import convolve2d as convolve2_scipy
from scipy.signal import fftconvolve

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] convolve2d(np.ndarray input,
                                                  np.ndarray filters,
                                                  np.ndarray biases,
                                                  int stride, int padding,
                                                  object activation_func,
                                                  bint use_bias=False):
    cdef int num_filters = filters.shape[0]
    cdef int filter_depth = filters.shape[1]
    cdef int filter_height = filters.shape[2]
    cdef int filter_width = filters.shape[3]
    cdef int input_depth = input.shape[0]
    cdef int input_height = input.shape[1]
    cdef int input_width = input.shape[2]

    if filter_depth != input_depth:
        raise ValueError("Filter depth must match input depth.")

    # Compute the dimensions of the output matrix
    cdef int output_height = ((input_height - filter_height + 2 * padding) // stride) + 1
    cdef int output_width = ((input_width - filter_width + 2 * padding) // stride) + 1
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((num_filters, output_height, output_width), dtype=np.float64)

    cdef int n, d, i, j
    cdef double conv_sum
    cdef np.ndarray[np.float64_t, ndim=2] conv_result

    # Apply padding to the input matrix if necessary
    cdef np.ndarray[np.float64_t, ndim=3] padded_input
    if padding > 0:
        padded_input = np.zeros((input_depth, input_height + 2 * padding, input_width + 2 * padding), dtype=np.float64)
        padded_input[:, padding:padding+input_height, padding:padding+input_width] = input
    else:
        padded_input = input

    for n in range(num_filters):
        conv_result = np.zeros((input_height, input_width), dtype=np.float64)

        for d in range(filter_depth):
            conv_result += fftconvolve(padded_input[d], filters[n, d], mode='same')

        for i in range(0, output_height):
            for j in range(0, output_width):
                conv_sum = conv_result[i * stride, j * stride]

                if use_bias and biases is not None:
                    conv_sum += biases[n]

                output[n, i, j] = conv_sum

            # apply the activation function
            output[n] = activation_func(output[n].flatten()).reshape(output_height, output_width)

    return output


cpdef tuple conv_backprop(np.ndarray delta,
                               np.ndarray[double, ndim=3] layer_input,
                               np.ndarray[double, ndim=4] kernels,
                               np.ndarray[double, ndim=1] biases,
                               object activation_derv,
                               bint use_bias):
    cdef int num_filters = kernels.shape[0]
    cdef int filter_depth = kernels.shape[1]
    cdef int i, j, d
    cdef np.ndarray kernel_gradient = np.zeros_like(kernels)
    cdef np.ndarray bias_gradient = np.zeros(num_filters) if use_bias else None
    cdef np.ndarray new_delta = np.zeros_like(layer_input)
    cdef np.ndarray rot_kernel, filtered

    if delta.ndim == 2:
        delta = delta.reshape(1, delta.shape[0], delta.shape[1])

    # Gradient calculation for each kernel
    for i in range(num_filters):
        for d in range(filter_depth):
            rot_kernel = np.rot90(kernels[i, d], 2)
            conv_result = convolve2_scipy(layer_input[d], rot_kernel, mode='valid')
            kernel_gradient[i, d] += np.sum(conv_result)

    # Bias gradients
    if use_bias:
        for i in range(num_filters):
            bias_gradient[i] = np.sum(delta[i])

    # Propagate delta to the previous layer
    for i in range(num_filters):
        for d in range(filter_depth):
            rot_kernel = np.rot90(kernels[i, d], 2)
            if delta.ndim == 3 and delta.shape[0] > i:
                delta_slice = delta[i]
                conv_result = convolve2_scipy(delta_slice, rot_kernel, mode='full')
                new_delta[d] += conv_result

    # Apply the activation derivative
    for i in range(new_delta.shape[0]):
        flat_delta = new_delta[i].flatten()
        flat_delta = activation_derv(flat_delta)
        new_delta[i] = flat_delta.reshape(new_delta[i].shape)

    return kernel_gradient, bias_gradient, new_delta







