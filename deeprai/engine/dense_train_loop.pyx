import numpy as np
cimport numpy as np
from deeprai.engine.cython import cython_optimizers as opti
from deeprai.engine.cython import activation as act
from deeprai.engine.cython import loss as loss
from deeprai.engine.cython.dense_operations import cython_back_propagate, cython_forward_propagate
from base_layer import DerivativeVals, WeightVals, BiasVals, NeuronVals

cpdef train(np.ndarray[np.float64_t, ndim=1] inputs, np.ndarray[np.float64_t, ndim=1] targets, int epochs,
            float learning_rate, float momentum, list activation_list, list activation_derv_list, list loss_function,
            bint verbose, int batch_size):
    cdef int start, end, num_batches
    cdef float sum_error
    cdef np.ndarray[np.float64_t, ndim=1] batch_inputs, batch_targets, output
    num_batches = len(inputs) // batch_size
    for epoch in range(epochs):
        sum_error = 0
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            for input, target in zip(batch_inputs, batch_targets):
                output = cython_back_propagate(input)
                cython_back_propagate(target - output)
                opti.gradient_descent(learning_rate)
                sum_error += loss.cython_mean_square_error(output, target)
        if verbose:
            print(f"Cost: {sum_error/(len(inputs))} at epoch{epoch +1}")
    print("Training complete!")

