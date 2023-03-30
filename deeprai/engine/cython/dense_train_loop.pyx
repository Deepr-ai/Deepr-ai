import numpy as np
cimport numpy as np
from deeprai.engine.cython import optimizers as opti
from alive_progress import alive_bar
from deeprai.engine.cython import activation as act
from deeprai.engine.cython import loss as loss
from deeprai.engine.cython.dense_operations import back_propagate, forward_propagate
from deeprai.engine.base_layer import DerivativeVals, WeightVals, NeuronVals, NetworkMetrics

cpdef train(np.ndarray[np.float64_t, ndim=2] inputs, np.ndarray[np.float64_t, ndim=2] targets,np.ndarray[np.float64_t, ndim=2] test_inputs,
            np.ndarray[np.float64_t, ndim=2] test_targets, int epochs,float learning_rate,
            float momentum, list activation_list, list activation_derv_list, list loss_function,list dropout_rate,
            list l2_penalty,list l1_penalty,bint early_stop, bint verbose, int batch_size):
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
    cdef float sum_error, error, total_rel_error, mean_rel_error, cur_acc
    cdef np.ndarray[np.float64_t, ndim=2] batch_inputs, batch_targets
    cdef np.ndarray[np.float64_t, ndim=1] output, abs_error, rel_error
    cdef float past_acc = .0
    num_batches = len(inputs) // batch_size
    cdef list neurons = NeuronVals.Neurons
    cdef list weights = WeightVals.Weights
    cdef list derv = DerivativeVals.Derivatives
    print("Starting Training...")
    for epoch in range(epochs):
        with alive_bar(num_batches, title=f"Epoch {epoch+1}", spinner="waves", dual_line=False) as bar:
            sum_error = 0
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_inputs = inputs[start:end]
                batch_targets = targets[start:end]
                for input, target in zip(batch_inputs, batch_targets):
                    output = forward_propagate(input, activation_list, neurons, weights, dropout_rate)
                    back_propagate(target - output, activation_derv_list, neurons, weights, derv, l1_penalty, l2_penalty)
                    opti.gradient_descent(learning_rate)
                    sum_error += loss_function[0](output, target)
                bar()
        error = 0
        #Testing model
        for input, target in zip(test_inputs, test_targets):
            output = forward_propagate(input, activation_list, neurons, weights, dropout_rate)
            abs_error = np.abs(output - target)
            rel_error = np.divide(abs_error, target, where=target != 0)
            mean_rel_error = np.mean(rel_error)*100
            error += mean_rel_error
            if early_stop:
                cur_acc= np.abs(100-total_rel_error)
                if cur_acc<past_acc:
                    print("Stopping due to val loss..")
                    return
                past_acc = cur_acc
        total_rel_error = error/len(test_inputs)
        accuracy = np.abs(100-total_rel_error)
        cost = sum_error/(len(inputs))
        NetworkMetrics[0].append(cost)
        NetworkMetrics[1].append(accuracy)
        NetworkMetrics[2].append(total_rel_error)
        NetworkMetrics[3].append(epoch+1)
        print(f"Epoch: {epoch+1} | Cost: {cost:4f} | Accuracy: {accuracy:.2f} | Relative Error: {total_rel_error:.3f}")
    print("Training complete!")

