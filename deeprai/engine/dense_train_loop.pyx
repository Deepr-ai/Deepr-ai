import numpy as np
cimport numpy as np
from deeprai.engine.cython import optimizers as opti
from deeprai.engine.cython import activation as act
from deeprai.engine.cython import loss as loss
from deeprai.engine.cython.dense_operations import back_propagate, forward_propagate

cpdef train(np.ndarray[np.float64_t, ndim=2] inputs, np.ndarray[np.float64_t, ndim=2] targets, int epochs,
            float learning_rate, float momentum, list activation_list, list activation_derv_list, list loss_function,
            bint verbose, int batch_size):
    cdef int start, end, num_batches
    cdef float sum_error
    cdef np.ndarray[np.float64_t, ndim=2] batch_inputs, batch_targets
    cdef np.ndarray[np.float64_t, ndim=1] output
    num_batches = len(inputs) // batch_size
    for epoch in range(epochs):
        sum_error = 0
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            for input, target in zip(batch_inputs, batch_targets):
                output = forward_propagate(input, activation_list)
                back_propagate(target - output, activation_derv_list)
                opti.gradient_descent(learning_rate)
                sum_error += loss_function[0](output, target)
        if verbose:
            print(f"Cost: {sum_error/(len(inputs))} at epoch: {epoch +1}")
    print("Training complete!")

