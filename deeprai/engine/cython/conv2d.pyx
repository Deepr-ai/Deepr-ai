import numpy as np

cpdef list compute_layer_valid(list layer_shape, list kernel_shape, list layer, list kernel):
    # 2d valid
    cdef int output_index
    cdef list output = []
    cdef list output_row
    for y_pos in range(layer_shape[1] - kernel_shape[1] + 1):
        # scoots down the matrix
        output_row = []
        for x_pos in range(layer_shape[0] - kernel_shape[0] + 1):
            # scoots across the matrix
            output_index = 0
            for kernel_y_level in range(kernel_shape[1]):
                # splits from into individual lists to sum together to get 1 output value
                output_index += np.dot(layer[y_pos + kernel_y_level][x_pos: x_pos + kernel_shape[1]],
                                       kernel[kernel_y_level])
            output_row.append(output_index)
        output.append(output_row)
    return output