import numpy as np
cimport numpy as np
from alive_progress import alive_bar
from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals, NetworkMetrics, KernelVals, KernalBiasVals
from deeprai.engine.cython.conv.cnn_operations import cnn_forward_prop, cnn_back_prop, evaluate
from deeprai.engine.cython.optimizers import gradient_descent_update, momentum_update, adagrad_update, rmsprop_update, \
    adam_update, adafactor_update
from deeprai.engine.cython.loss import categorical_cross_entropy, mean_square_error, mean_absolute_error

cpdef train(np.ndarray inputs, np.ndarray targets,
            np.ndarray test_inputs, np.ndarray test_targets, int epochs,
            float learning_rate, str loss_function, list dropout_rate, list activation_derv_list, list l2_penalty, list l1_penalty, bint use_bias, bint verbose, int batch_size, str optimizer_name,
            list action_list, list action_list_str, list pool_size):
    cdef int inputs_len = inputs.shape[0]
    cdef int test_inputs_len = test_inputs.shape[0]
    cdef int num_batches = inputs_len // batch_size

    if optimizer_name == "adam":
        kernel_m = [np.zeros_like(w) for w in KernelVals.Kernels]
        kernel_v = [np.zeros_like(w) for w in KernelVals.Kernels]
        bias_m = [np.zeros_like(b) for b in KernalBiasVals.KernalBias]
        bias_v = [np.zeros_like(b) for b in KernalBiasVals.KernalBias]
        dense_m = [np.zeros_like(w) for w in WeightVals.Weights]
        dense_v = [np.zeros_like(w) for w in WeightVals.Weights]
        dense_bias_m = [np.zeros_like(b) for b in BiasVals.Biases]
        dense_bias_v = [np.zeros_like(b) for b in BiasVals.Biases]
        t = 0  # Initialize timestep for Adam

    elif optimizer_name == "momentum" or optimizer_name == "rmsprop":
        kernel_velocity_optimizer = [np.zeros_like(k) for k in KernelVals.Kernels]
        kernel_bias_velocity_optimizer = [np.zeros_like(b) for b in KernalBiasVals.KernalBias]
        dense_velocity_optimizer = [np.zeros_like(w) for w in WeightVals.Weights]
        dense_bias_velocity_optimizer = [np.zeros_like(b) for b in BiasVals.Biases]

    elif optimizer_name == "adagrad":
        # Very confusing names
        kernel_accumulated_grad = [np.zeros_like(k, dtype=np.float64) for k in KernelVals.Kernels]
        kernel_bias_accumulated_grad = [np.zeros_like(b, dtype=np.float64) for b in KernalBiasVals.KernalBias]
        dense_accumulated_grad = [np.zeros_like(w, dtype=np.float64) for w in WeightVals.Weights]
        dense_bias_accumulated_grad = [np.zeros_like(b, dtype=np.float64) for b in BiasVals.Biases]

    elif optimizer_name == "adafactor":
        dense_squared_accumulators = [np.zeros_like(w) for w in WeightVals.Weights]
        dense_bias_squared_accumulators = [np.zeros_like(b) for b in BiasVals.Biases]
        kernel_squared_accumulators = [np.zeros_like(w) for w in WeightVals.Weights]
        kernel_bias_squared_accumulators = [np.zeros_like(b) for b in BiasVals.Biases]


    for epoch in range(epochs):
        sum_error = 0.0
        with alive_bar(num_batches, title=f"Epoch {epoch + 1}", spinner="waves", dual_line=False) as bar:
            for batch_start in range(0, inputs_len, batch_size):
                batch_end = min(batch_start + batch_size, inputs_len)
                batch_inputs = inputs[batch_start:batch_end]
                batch_targets = targets[batch_start:batch_end]

                kernel_weight_gradients_accumulated = None
                kernel_bias_gradients_accumulated = None
                dense_weight_gradients_accumulated = None
                dense_bias_gradients_accumulated = None
                batch_count = 0

                for single_input, single_target in zip(batch_inputs, batch_targets):
                    output, layer_outputs = cnn_forward_prop(single_input, action_list, action_list_str, True)
                    if loss_function == "cross entropy":
                        sum_error += categorical_cross_entropy(output, single_target)
                    elif loss_function == "mean square error":
                        sum_error += mean_square_error(output, single_target)
                    elif loss_function == "mean absolute error":
                        sum_error += mean_absolute_error(output, single_target)
                    else:
                        raise ValueError(f"Unsupported loss type: {loss_function}")
                    kernel_weight_gradients, kernel_bias_gradients, dense_weight_gradients, dense_bias_gradients = cnn_back_prop(
                        final_output=output,
                        true_output=single_target,
                        layer_outputs=layer_outputs,
                        steps=action_list_str,
                        activation_derv_list=activation_derv_list,
                        conv_kernels=KernelVals.Kernels,
                        dense_weights=WeightVals.Weights,
                        conv_biases=KernalBiasVals.KernalBias,
                        dense_biases=BiasVals.Biases,
                        l1_penalty=l1_penalty,
                        l2_penalty=l2_penalty,
                        use_bias=use_bias,
                        loss_type=loss_function,
                        pool_size = pool_size
                    )

                    if kernel_weight_gradients_accumulated is None:
                        kernel_weight_gradients_accumulated = [np.zeros_like(grad) for grad in kernel_weight_gradients]
                        kernel_bias_gradients_accumulated = [np.zeros_like(grad) for grad in kernel_bias_gradients]
                        dense_weight_gradients_accumulated = [np.zeros_like(grad) for grad in dense_weight_gradients]
                        dense_bias_gradients_accumulated = [np.zeros_like(grad) for grad in dense_bias_gradients]

                    for layer in range(len(kernel_weight_gradients)):
                        kernel_weight_gradients_accumulated[layer] += kernel_weight_gradients[layer]
                        kernel_bias_gradients_accumulated[layer] += kernel_bias_gradients[layer]

                    for layer in range(len(dense_weight_gradients)):
                        dense_weight_gradients_accumulated[layer] += dense_weight_gradients[layer]
                        dense_bias_gradients_accumulated[layer] += dense_bias_gradients[layer]

                    batch_count += 1

                for layer in range(len(kernel_weight_gradients_accumulated)):
                    kernel_weight_gradients_accumulated[layer] = kernel_weight_gradients_accumulated[layer].astype(
                        np.float64) / batch_count
                    kernel_bias_gradients_accumulated[layer] = kernel_bias_gradients_accumulated[layer].astype(
                        np.float64) / batch_count

                for layer in range(len(dense_weight_gradients_accumulated)):
                    dense_weight_gradients_accumulated[layer] = dense_weight_gradients_accumulated[layer].astype(
                        np.float64) / batch_count
                    dense_bias_gradients_accumulated[layer] = dense_bias_gradients_accumulated[layer].astype(
                        np.float64) / batch_count


                # Update perams
                if optimizer_name == "gradient descent":

                    KernelVals.Kernels = gradient_descent_update(KernelVals.Kernels, kernel_weight_gradients_accumulated,
                                                                 learning_rate)
                    WeightVals.Weights = gradient_descent_update(WeightVals.Weights, dense_weight_gradients_accumulated,
                                                                 learning_rate)
                    if use_bias:
                        BiasVals.Biases = gradient_descent_update(BiasVals.Biases, dense_bias_gradients_accumulated,
                                                                  learning_rate)
                        KernalBiasVals.KernalBias = gradient_descent_update(KernalBiasVals.KernalBias,
                                                                            kernel_bias_gradients_accumulated,
                                                                            learning_rate)

                elif optimizer_name == "momentum":
                    KernelVals.Kernels, kernel_velocity_optimizer = momentum_update(KernelVals.Kernels, kernel_weight_gradients,
                                                                          kernel_velocity_optimizer, learning_rate)
                    WeightVals.Weights, dense_velocity_optimizer = momentum_update(WeightVals.Weights, dense_weight_gradients_accumulated,
                                                                         dense_velocity_optimizer, learning_rate)
                    if use_bias:
                        KernalBiasVals.KernalBias, kernel_bias_velocity_optimizer = momentum_update(KernalBiasVals.KernalBias, kernel_bias_gradients_accumulated,
                                                                                          kernel_bias_velocity_optimizer, learning_rate)
                        BiasVals.Biases, dense_bias_velocity_optimizer = momentum_update(BiasVals.Biases, dense_bias_gradients_accumulated,
                                                                               dense_bias_velocity_optimizer, learning_rate)

                elif optimizer_name == "adam":
                    t += 1
                    KernelVals.Kernels, kernel_m, kernel_v = adam_update(
                        KernelVals.Kernels, kernel_weight_gradients_accumulated, kernel_m, kernel_v,
                        learning_rate, t=t)
                    WeightVals.Weights, dense_m, dense_v = adam_update(
                        WeightVals.Weights, dense_weight_gradients_accumulated, dense_m, dense_v,
                        learning_rate, t=t)

                    if use_bias:
                        BiasVals.Biases, dense_bias_m, dense_bias_v = adam_update(
                            BiasVals.Biases, dense_bias_gradients_accumulated, dense_bias_m, dense_bias_v,
                            learning_rate, t=t)
                        KernalBiasVals.KernalBias, bias_m, bias_v = adam_update(
                            KernalBiasVals.KernalBias, kernel_bias_gradients_accumulated, bias_m, bias_v,
                            learning_rate, t=t)

                elif optimizer_name == "rmsprop":
                    KernelVals.Kernels, kernel_velocity_optimizer = rmsprop_update(KernelVals.Kernels, kernel_weight_gradients_accumulated,
                                                                          kernel_velocity_optimizer, learning_rate)
                    WeightVals.Weights, dense_velocity_optimizer = rmsprop_update(WeightVals.Weights, dense_weight_gradients_accumulated,
                                                                         dense_velocity_optimizer, learning_rate)
                    if use_bias:
                        KernalBiasVals.KernalBias, kernel_bias_velocity_optimizer = rmsprop_update(KernalBiasVals.KernalBias, kernel_bias_gradients_accumulated,
                                                                                          kernel_bias_velocity_optimizer, learning_rate)
                        BiasVals.Biases, dense_bias_velocity_optimizer = rmsprop_update(BiasVals.Biases, dense_bias_gradients_accumulated,
                                                                               dense_bias_velocity_optimizer, learning_rate)

                elif optimizer_name == "adagrad":
                    KernelVals.Kernels, kernel_accumulated_grad = adagrad_update(KernelVals.Kernels, kernel_weight_gradients_accumulated, kernel_accumulated_grad, learning_rate)
                    WeightVals.Weights, dense_accumulated_grad = adagrad_update(WeightVals.Weights, dense_weight_gradients_accumulated,dense_accumulated_grad, learning_rate)
                    if use_bias:
                        KernalBiasVals.KernalBias, kernel_bias_accumulated_grad = adagrad_update(KernalBiasVals.KernalBias, kernel_bias_gradients_accumulated, kernel_bias_accumulated_grad,learning_rate)
                        BiasVals.Biases, dense_bias_accumulated_grad = adagrad_update(BiasVals.Biases, dense_bias_gradients_accumulated, dense_bias_accumulated_grad, learning_rate)

                elif optimizer_name == "adafactor":
                    KernelVals.Kernels, kernel_squared_accumulators = adafactor_update(KernelVals.Kernels, kernel_weight_gradients_accumulated, kernel_squared_accumulators, learning_rate)
                    WeightVals.Weights, dense_squared_accumulators = adafactor_update(WeightVals.Weights, dense_weight_gradients_accumulated, dense_squared_accumulators, learning_rate)
                    if use_bias:
                        KernalBiasVals.KernalBias, kernel_bias_squared_accumulators = adafactor_update(KernalBiasVals.KernalBias, kernel_bias_gradients_accumulated, kernel_bias_squared_accumulators, learning_rate)
                        BiasVals.Biases, dense_bias_squared_accumulators = adafactor_update(BiasVals.Biases, dense_bias_gradients_accumulated, dense_bias_squared_accumulators, learning_rate)

                bar()

        network_report = evaluate(test_inputs, test_targets, action_list, action_list_str, loss_function)

        NetworkMetrics[0].append(network_report['cost'])
        NetworkMetrics[1].append(network_report['accuracy'])
        NetworkMetrics[2].append(network_report['relative_error'])
        NetworkMetrics[3].append(epoch + 1)
        if verbose:
            print(
                f"Epoch: {epoch + 1} | Cost: {np.float64(network_report['cost']):.4f} | Accuracy: {np.float64(network_report['accuracy']):.2f}% | Relative Error: {np.float64(network_report['relative_error']):.3f}%"
            )