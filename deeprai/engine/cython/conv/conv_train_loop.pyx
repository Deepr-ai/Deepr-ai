import numpy as np
cimport numpy as np
from alive_progress import alive_bar
from deeprai.engine.base_layer import NeuronVals, WeightVals, BiasVals, NetworkMetrics, KernelVals, KernalBiasVals
from deeprai.engine.cython.conv.cnn_operations import cnn_forward_prop, cnn_back_prop
from deeprai.engine.cython.optimizers import gradient_descent_update, momentum_update, adagrad_update, rmsprop_update, \
    adam_update, adadelta_update, adafactor_update
from deeprai.engine.cython.loss import categorical_cross_entropy, mean_square_error, mean_absolute_error

cpdef train(np.ndarray inputs, np.ndarray targets,
            np.ndarray test_inputs, np.ndarray test_targets, int epochs,
            float learning_rate, str loss_function, list dropout_rate, list activation_derv_list, list l2_penalty, list l1_penalty, bint use_bias, bint verbose, int batch_size, str optimizer_name,
            list action_list, list action_list_str):
    cdef int inputs_len = inputs.shape[0]
    cdef int test_inputs_len = test_inputs.shape[0]
    cdef int num_batches = inputs_len // batch_size

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
                    sum_error += mean_square_error(output, single_target)
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
                        loss_type="mean square error"
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

                KernelVals.Kernels = gradient_descent_update(KernelVals.Kernels, kernel_weight_gradients_accumulated,
                                                             learning_rate)
                KernalBiasVals.KernalBias = gradient_descent_update(KernalBiasVals.KernalBias,

                                                                kernel_bias_gradients_accumulated, learning_rate)
                WeightVals.Weights = gradient_descent_update(WeightVals.Weights, dense_weight_gradients_accumulated,
                                                             learning_rate)
                BiasVals.Biases = gradient_descent_update(BiasVals.Biases, dense_bias_gradients_accumulated,
                                                          learning_rate)

                bar()
            test_inputs_len = len(test_inputs)
            sum_error = 0
            # Initialize arrays for absolute and relative errors
            abs_errors = np.zeros(test_inputs_len)
            rel_errors = np.zeros(test_inputs_len)
        for i, (input_val, target) in enumerate(zip(test_inputs, test_targets)):
            # Forward propagation on the test input
            output = cnn_forward_prop(input_val, action_list)
            # Calculate absolute error
            abs_error = np.abs(output - target)
            rel_error = np.divide(abs_error, target, where=target != 0)

            # Store the errors
            abs_errors[i] = np.sum(abs_error)
            rel_errors[i] = np.mean(rel_error) * 100
            sum_error += np.sum(abs_error)
        mean_rel_error = np.mean(rel_errors)
        total_rel_error = np.sum(rel_errors) / test_inputs_len
        accuracy = np.abs(100 - total_rel_error)

        NetworkMetrics[0].append(sum_error / test_inputs_len)
        NetworkMetrics[1].append(accuracy)
        NetworkMetrics[2].append(total_rel_error)
        NetworkMetrics[3].append(epoch + 1)

        if verbose:
            print(
                f"Epoch: {epoch + 1} | Cost: {sum_error / test_inputs_len:.4f} | Accuracy: {accuracy:.2f}% | Relative Error: {total_rel_error:.3f}%")






