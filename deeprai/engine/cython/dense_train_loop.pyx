import numpy as np
cimport numpy as np
from alive_progress import alive_bar

from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals, NetworkMetrics
from deeprai.engine.cython.dense_operations import forward_propagate, back_propagate, evaluate
from deeprai.engine.cython.optimizers import gradient_descent_update, momentum_update, adagrad_update, rmsprop_update, \
    adam_update, adafactor_update
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
    cdef dict network_report

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

    elif optimizer_name == "adafactor":
        weight_v = [np.zeros_like(w) for w in WeightVals.Weights]
        bias_v = [np.zeros_like(b) for b in BiasVals.Biases]

    for epoch in range(epochs):
        sum_error = 0.0
        with alive_bar(num_batches+1, title=f"Epoch {epoch + 1}", spinner="waves", dual_line=False) as bar:
            for batch_start in range(0, inputs_len, batch_size):
                batch_end = min(batch_start + batch_size, inputs_len)
                batch_inputs = inputs[batch_start:batch_end]
                batch_targets = targets[batch_start:batch_end]

                weight_gradients_accumulated = None
                bias_gradients_accumulated = None
                batch_count = 0

                for single_input, single_target in zip(batch_inputs, batch_targets):
                    # Forward Propagation
                    output = forward_propagate(single_input, activation_list, NeuronVals.Neurons, WeightVals.Weights,
                                                BiasVals.Biases, use_bias, dropout_rate)
                    # Calculate error
                    if loss_function[0] == "cross entropy":
                        sum_error += categorical_cross_entropy(output, single_target)
                    elif loss_function[0] == "mean square error":
                        sum_error += mean_square_error(output, single_target)
                    elif loss_function[0] == "mean absolute error":
                        sum_error += mean_absolute_error(output, single_target)
                    else:
                        raise ValueError(f"Unsupported loss type: {loss_function}")
                    weight_gradients, bias_gradients = back_propagate(output, single_target, activation_derv_list,
                                                                      NeuronVals.Neurons, WeightVals.Weights,
                                                                      l1_penalty, l2_penalty, use_bias,
                                                                      loss_function[0])
                    # Accumulate gradients
                    if weight_gradients_accumulated is None:
                        weight_gradients_accumulated = weight_gradients
                        bias_gradients_accumulated = bias_gradients
                    else:
                        for layer in range(len(weight_gradients)):
                            weight_gradients_accumulated[layer] += weight_gradients[layer]
                            bias_gradients_accumulated[layer] += bias_gradients[layer]

                    batch_count += 1

                # Average the accumulated gradients
                for layer in range(len(weight_gradients_accumulated)):
                    weight_gradients_accumulated[layer] /= batch_count
                    bias_gradients_accumulated[layer] /= batch_count

                # Update Weights and Biases
                if optimizer_name == "gradient descent":
                    WeightVals.Weights = gradient_descent_update(WeightVals.Weights, weight_gradients, learning_rate)
                    if use_bias:
                        BiasVals.Biases = gradient_descent_update(BiasVals.Biases, bias_gradients,learning_rate)
                elif optimizer_name == "momentum":
                    WeightVals.Weights, weight_velocity = momentum_update(
                        WeightVals.Weights,weight_gradients, weight_velocity, learning_rate, momentum)
                    if use_bias:
                        BiasVals.Biases, bias_velocity = momentum_update(
                            BiasVals.Biases, bias_gradients, bias_velocity, learning_rate, momentum)

                elif optimizer_name == "adagrad":
                    WeightVals.Weights, weight_accumulated_grad = adagrad_update(WeightVals.Weights, weight_gradients,
                                                                                 weight_accumulated_grad, learning_rate)

                    if use_bias:
                        BiasVals.Biases, bias_accumulated_grad = adagrad_update(BiasVals.Biases,
                                                                                     bias_gradients,
                                                                                     bias_accumulated_grad,
                                                                                     learning_rate)
                elif optimizer_name == "rmsprop":
                    WeightVals.Weights, weight_velocity = rmsprop_update(
                        WeightVals.Weights,weight_gradients, weight_velocity, learning_rate, momentum)
                    if use_bias:
                        BiasVals.Biases, bias_velocity = rmsprop_update(
                            BiasVals.Biases, bias_gradients, bias_velocity, learning_rate, momentum)

                elif optimizer_name == "adam":
                    t += 1
                    WeightVals.Weight, weight_m, weight_v,  = adam_update(
                        WeightVals.Weights, weight_gradients_accumulated, weight_m, weight_v,
                        learning_rate, t=t)
                    if use_bias:
                        BiasVals.Biases, bias_m, bias_v,  = adam_update(
                            BiasVals.Biases, bias_gradients_accumulated, bias_m, bias_v,
                            learning_rate, t=t)

                elif optimizer_name == "adafactor":
                    WeightVals.Weights, BiasVals.Biases, weight_v, bias_v = adafactor_update(
                        WeightVals.Weights, BiasVals.Biases, weight_gradients, bias_gradients,
                        weight_v, bias_v, learning_rate=learning_rate, use_bias=use_bias)

                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

                if epoch % checkpoint_interval == 0 and checkpoint_dir_location is not None:
                    Save(f"{checkpoint_dir_location}/checkpoint_epoch_{epoch}.deepr")

                bar()  # update the progress bar after batch

        network_report = evaluate(test_inputs, test_targets, activation_list, use_bias, dropout_rate,
                                  loss_function[0])


        NetworkMetrics[0].append(network_report['cost'])
        NetworkMetrics[1].append(network_report['accuracy'])
        NetworkMetrics[2].append(network_report['relative_error'])
        NetworkMetrics[3].append(epoch + 1)


        if verbose:
            print(
                f"Epoch: {epoch + 1} | Cost: {np.float64(network_report['cost']):.4f} | Accuracy: {np.float64(network_report['accuracy']):.2f}% | Relative Error: {np.float64(network_report['relative_error']):.3f}%"
            )
