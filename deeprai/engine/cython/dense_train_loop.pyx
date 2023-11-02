import numpy as np
cimport numpy as np
from alive_progress import alive_bar

from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals, NetworkMetrics
from deeprai.engine.cython.dense_operations import forward_propagate, back_propagate
from deeprai.engine.cython.optimizers import gradient_descent_update, momentum_update, adagrad_update, rmsprop_update, \
    adam_update, adadelta_update, adafactor_update
from deeprai.engine.cython.loss import categorical_cross_entropy, mean_square_error, mean_absolute_error
from deeprai.tools.file_manager.save import Save

cpdef train(np.ndarray[np.float64_t, ndim=2] inputs, np.ndarray[np.float64_t, ndim=2] targets,
            np.ndarray[np.float64_t, ndim=2] test_inputs, np.ndarray[np.float64_t, ndim=2] test_targets, int epochs,
            float learning_rate, float momentum, list activation_list, list activation_derv_list, list loss_function,
            list dropout_rate, list l2_penalty, list l1_penalty, bint use_bias, bint verbose, int batch_size, str optimizer_name,
            int checkpoint_interval, str checkpoint_dir_location=None):

    cdef int inputs_len = inputs.shape[0]
    cdef int test_inputs_len = test_inputs.shape[0]
    cdef int num_batches = inputs_len // batch_size

    if optimizer_name == "momentum":
        weight_velocity = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_velocity = [np.zeros_like(b) for b in BiasVals.Biases]

    elif optimizer_name == "adagrad":
        weight_accumulated_grad = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_accumulated_grad = [np.zeros_like(b) for b in BiasVals.Biases]

    elif optimizer_name == "rmsprop":
        weight_v = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_v = [np.zeros_like(b) for b in BiasVals.Biases]

    elif optimizer_name == "adam":
        weight_m = [np.zeros_like(w) for w in WeightVals.Weights]
        weight_v = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_m = [np.zeros_like(b) for b in BiasVals.Biases]
        bias_v = [np.zeros_like(b) for b in BiasVals.Biases]
        t = 0  # Initialize timestep for Adam

    elif optimizer_name == "adadelta":
        weight_accumulated_grad = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_accumulated_grad = [np.zeros_like(b) for b in BiasVals.Biases]
        weight_delta_accumulated = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_delta_accumulated = [np.zeros_like(b) for b in BiasVals.Biases]

    elif optimizer_name == "adafactor":
        weight_v = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_v = [np.zeros_like(b) for b in BiasVals.Biases]

    for epoch in range(epochs):
        sum_error = 0.0
        with alive_bar(num_batches + 1, title=f"Epoch {epoch + 1}", spinner="waves", dual_line=False) as bar:
            for batch_start in range(0, inputs_len, batch_size):
                batch_end = min(batch_start + batch_size, inputs_len)
                batch_inputs = inputs[batch_start:batch_end]
                batch_targets = targets[batch_start:batch_end]

                for single_input, single_target in zip(batch_inputs, batch_targets):
                    # Forward Propagation
                    outputs = forward_propagate(single_input, activation_list, NeuronVals.Neurons, WeightVals.Weights,
                                                BiasVals.Biases, use_bias, dropout_rate)
                    # Calculate error
                    if loss_function[0] == "cross entropy":
                        sum_error += categorical_cross_entropy(outputs, single_target)
                    elif loss_function[0] == "mean square error":
                        sum_error += mean_square_error(outputs, single_target)
                    elif loss_function[0] == "mean absolute error":
                        sum_error += mean_absolute_error(outputs, single_target)
                    else:
                        raise ValueError(f"Unsupported loss type: {loss_function}")
                    weight_gradients, bias_gradients = back_propagate(outputs, single_target, activation_derv_list,
                                                                      NeuronVals.Neurons, WeightVals.Weights,
                                                                      l1_penalty,
                                                                      l2_penalty, use_bias, loss_function[0])

                    # Update Weights and Biases
                    if optimizer_name == "gradient descent":
                        WeightVals.Weights, BiasVals.Biases = gradient_descent_update(WeightVals.Weights,
                                                                                      BiasVals.Biases, weight_gradients,
                                                                                      bias_gradients, learning_rate,
                                                                                      use_bias)
                    elif optimizer_name == "momentum":
                        WeightVals.Weights, BiasVals.Biases, weight_velocity, bias_velocity = momentum_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients, weight_velocity,
                            bias_velocity, learning_rate, momentum, use_bias)

                    elif optimizer_name == "adagrad":
                        WeightVals.Weights, BiasVals.Biases, weight_accumulated_grad, bias_accumulated_grad = adagrad_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients,
                            weight_accumulated_grad, bias_accumulated_grad, learning_rate, epsilon, use_bias)

                    elif optimizer_name == "rmsprop":
                        #Temp values .18 for kwargs
                        beta = 0.9
                        epsilon = 1e-7
                        WeightVals.Weights, BiasVals.Biases, weight_v, bias_v = rmsprop_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients, weight_v, bias_v,
                            learning_rate, beta, epsilon, use_bias)

                    elif optimizer_name == "adam":
                        t += 1
                        WeightVals.Weights, BiasVals.Biases, weight_m, weight_v, bias_m, bias_v = adam_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients, weight_m, weight_v,
                            bias_m, bias_v, learning_rate, t=t, use_bias=use_bias)

                    elif optimizer_name == "adadelta":
                        WeightVals.Weights, BiasVals.Biases, weight_accumulated_grad, bias_accumulated_grad, weight_delta_accumulated, bias_delta_accumulated = adadelta_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients,
                            weight_accumulated_grad,
                            bias_accumulated_grad, weight_delta_accumulated, bias_delta_accumulated, use_bias=use_bias)

                    elif optimizer_name == "adafactor":
                        WeightVals.Weights, BiasVals.Biases, weight_v, bias_v = adafactor_update(
                            WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients,
                            weight_v, bias_v, learning_rate=learning_rate, use_bias=use_bias)

                    else:
                        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

                    if epoch % checkpoint_interval == 0 and checkpoint_dir_location is not None:
                        Save(f"{checkpoint_dir_location}/checkpoint_epoch_{epoch}.deepr")

                bar()  # update the progress bar after batch
        error = 0
        abs_errors = np.zeros(test_inputs_len)
        rel_errors = np.zeros(test_inputs_len)

        for i, (input_val, target) in enumerate(zip(test_inputs, test_targets)):
            output = forward_propagate(input_val, activation_list, NeuronVals.Neurons, WeightVals.Weights, BiasVals.Biases, use_bias, dropout_rate)
            abs_error = np.abs(output - target)
            rel_error = np.divide(abs_error, target, where=target != 0)

            abs_errors[i] = np.sum(abs_error)
            rel_errors[i] = np.mean(rel_error) * 100

        mean_rel_error = np.mean(rel_errors)
        total_rel_error = np.sum(rel_errors) / test_inputs_len
        accuracy = np.abs(100 - total_rel_error)
        NetworkMetrics[0].append(sum_error / inputs_len)
        NetworkMetrics[1].append(accuracy)
        NetworkMetrics[2].append(total_rel_error)
        NetworkMetrics[3].append(epoch + 1)

        if verbose:
            print(f"Epoch: {epoch + 1} | Cost: {sum_error / inputs_len:.4f} | Accuracy: {accuracy:.2f}% | Relative Error: {total_rel_error:.3f}%")
