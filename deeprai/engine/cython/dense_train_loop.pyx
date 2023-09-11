import numpy as np
cimport numpy as np
from deeprai.engine.cython import optimizers as opti
from alive_progress import alive_bar
from deeprai.engine.cython import activation as act
from deeprai.engine.cython import loss as loss
from deeprai.engine.cython.dense_operations import back_propagate, forward_propagate
from deeprai.engine.base_layer import DerivativeVals, WeightVals, NeuronVals, NetworkMetrics, LossString

cpdef train(np.ndarray[np.float64_t, ndim=2] inputs, np.ndarray[np.float64_t, ndim=2] targets,
            np.ndarray[np.float64_t, ndim=2] test_inputs,
            np.ndarray[np.float64_t, ndim=2] test_targets, int epochs, float learning_rate,
            float momentum, list activation_list, list activation_derv_list, list loss_function, list dropout_rate,
            list l2_penalty, list l1_penalty, bint early_stop, bint verbose, int batch_size):
    cdef int start, end, num_batches
    cdef float sum_error, error, total_rel_error, mean_rel_error, cur_acc
    cdef np.ndarray[np.float64_t, ndim=2] batch_inputs, batch_targets
    cdef np.ndarray[np.float64_t, ndim=1] output, abs_error, rel_error
    cdef float past_acc = .0
    num_batches = len(inputs) // batch_size
    cdef list neurons = NeuronVals.Neurons
    cdef list weights = WeightVals.Weights
    cdef list derv = DerivativeVals.Derivatives
    cdef int test_inputs_len = len(test_inputs)
    cdef int inputs_len = len(inputs)

    print("Starting Training...")
    for epoch in range(epochs):
        with alive_bar(num_batches, title=f"Epoch {epoch + 1}", spinner="waves", dual_line=False) as bar:
            sum_error = 0
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_inputs = inputs[start:end]
                batch_targets = targets[start:end]

                # Initialize batch errors to zero
                batch_errors = np.zeros(batch_size)

                for i, (input, target) in enumerate(zip(batch_inputs, batch_targets)):
                    output = forward_propagate(input, activation_list, neurons, weights, dropout_rate)
                    back_propagate(target - output, activation_derv_list, neurons, weights, derv, l1_penalty,
                                   l2_penalty)
                    opti.gradient_descent(learning_rate)

                    # Accumulate errors for each input in the batch
                    batch_errors[i] = loss_function[0](output, target)

                # Calculate the sum error for the batch
                sum_error += np.sum(batch_errors)

                bar()
        error = 0
        abs_errors = np.zeros(test_inputs_len)
        rel_errors = np.zeros(test_inputs_len)

        for i, (input, target) in zip(range(test_inputs_len), zip(test_inputs, test_targets)):
            output = forward_propagate(input, activation_list, neurons, weights, dropout_rate)
            abs_error = np.abs(output - target)
            rel_error = np.divide(abs_error, target, where=target != 0)

            abs_errors[i] = np.sum(abs_error)
            rel_errors[i] = np.mean(rel_error) * 100

        mean_rel_error = np.mean(rel_errors)
        total_rel_error = np.sum(rel_errors) / test_inputs_len
        accuracy = np.abs(100 - total_rel_error)
        cost = sum_error / inputs_len
        NetworkMetrics[0].append(cost)
        NetworkMetrics[1].append(accuracy)
        NetworkMetrics[2].append(total_rel_error)
        NetworkMetrics[3].append(epoch + 1)
        print(
            f"Epoch: {epoch + 1} | Cost: {cost:4f} | Accuracy: {accuracy:.2f} | Relative Error: {total_rel_error:.3f}")
        if early_stop:
            cur_acc = np.abs(100 - total_rel_error)
            if cur_acc < past_acc:
                print("Stopping due to val loss..")
                return
            past_acc = cur_acc
    print("Training complete!")
