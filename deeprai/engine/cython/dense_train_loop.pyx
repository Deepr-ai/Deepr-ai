import numpy as np
cimport numpy as np
from deeprai.engine.cython import optimizers as opti
from alive_progress import alive_bar
from deeprai.engine.cython import activation as act
from deeprai.engine.cython import loss as loss
from deeprai.engine.cython.dense_operations import back_propagate, forward_propagate
from deeprai.engine.base_layer import DerivativeVals, WeightVals, NeuronVals

cpdef train(np.ndarray[np.float64_t, ndim=2] inputs, np.ndarray[np.float64_t, ndim=2] targets, int epochs,
            float learning_rate, float momentum, list activation_list, list activation_derv_list, list loss_function,
            bint verbose, int batch_size):
    """
     Trains a neural network model using gradient descent optimization.

     Parameters:
     -----------
     inputs: np.ndarray[np.float64_t, ndim=2]
         Input features to the neural network.
     targets: np.ndarray[np.float64_t, ndim=2]
         Target values for the model to learn.
     epochs: int
         Number of iterations over the entire input data.
     learning_rate: float
         Step size for gradient descent optimization.
     momentum: float
         Momentum term for the optimization.
     activation_list: list
         List of activation functions to use in the network.
     activation_derv_list: list
         List of activation functions derivatives to use in backpropagation.
     loss_function: list
         Loss function to use for computing error.
     verbose: bool
         Whether to display progress bar or not.
     batch_size: int
         Number of input samples per forward-backward pass.

     Returns:
     --------
     None
     """
    cdef int start, end, num_batches
    cdef float sum_error
    cdef np.ndarray[np.float64_t, ndim=2] batch_inputs, batch_targets
    cdef np.ndarray[np.float64_t, ndim=1] output
    num_batches = len(inputs) // batch_size
    cdef list neurons = NeuronVals.Neurons
    cdef list weights = WeightVals.Weights
    cdef list derv = DerivativeVals.Derivatives
    with alive_bar(epochs, dual_line=True, title="Training", spinner="waves") as bar:
        for epoch in range(epochs):
            sum_error = 0
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_inputs = inputs[start:end]
                batch_targets = targets[start:end]
                for input, target in zip(batch_inputs, batch_targets):
                    output = forward_propagate(input, activation_list, neurons, weights)
                    back_propagate(target - output, activation_derv_list, neurons, weights, derv)
                    opti.gradient_descent(learning_rate)
                    sum_error += loss_function[0](output, target)
            if verbose:
                bar.text = f"Cost: {sum_error/(len(inputs))}"
                bar()
    print("Training complete!")

