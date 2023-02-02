import numpy as np
cimport numpy as np
from deeprai.engine.cython import activation as act
from deeprai.engine.base_layer import DerivativeVals, WeightVals, NeuronVals

derv = DerivativeVals.Derivatives
neurons = NeuronVals.Neurons
weights = WeightVals.Weights
#PUBLIC FUNCTIONprint(models.run(x))
cpdef np.ndarray[np.float64_t, ndim=1] forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs, list activation_list):
    NeuronVals.Neurons[0]= inputs
    # activation_list -> a list of lambda functions
    cdef np.ndarray[np.float64_t, ndim = 1] layer_outputs
    for layer, weight in enumerate(WeightVals.Weights):
        layer_outputs = np.dot(NeuronVals.Neurons[layer], weight)
        NeuronVals.Neurons[layer+1] = activation_list[layer](layer_outputs)
    return NeuronVals.Neurons[-1]

cpdef np.ndarray[np.float64_t, ndim=1] back_propagate(np.ndarray[np.float64_t, ndim=1] loss,  list activation_derv_list):
    cdef np.ndarray[np.float64_t, ndim = 1] delta
    cdef np.ndarray[np.float64_t, ndim = 2] delta_reshape, current_reshaped
    for layer in reversed(range(len(DerivativeVals.Derivatives))):
        delta = loss * activation_derv_list[layer](NeuronVals.Neurons[layer+1])
        delta_reshape = delta.reshape(delta.shape[0], -1).T
        current_reshaped = NeuronVals.Neurons[layer].reshape(NeuronVals.Neurons[layer].shape[0], -1)
        DerivativeVals.Derivatives[layer] = np.dot(current_reshaped, delta_reshape)
        loss = np.dot(delta, WeightVals.Weights[layer].T)

#PRIVATE FUNCTION
#note to self, add bies gradent
cdef np.ndarray[np.float64_t, ndim=1] cython_back_propagate(np.ndarray[np.float64_t, ndim=1] loss,  list activation_derv_list ):
    cdef np.ndarray[np.float64_t, ndim = 1] delta, delta_reshape, current_reshaped
    for layer in reversed(range(len(derv))):
        delta = loss * activation_derv_list[layer](neurons[layer+1])
        delta_reshape = delta.reshape(delta.shape[0], -1).T
        current_reshaped = neurons[layer].reshape(neurons[layer].shape[0], -1)
        derv[layer] = np.dot(current_reshaped, delta_reshape)
        loss = np.dot(delta, weights[layer].T)


cdef np.ndarray[np.float64_t, ndim=1] cython_forward_propagate(np.ndarray[np.float64_t, ndim=1] inputs, list activation_list):
    neurons[0] = inputs
    # activation_list -> a list of lambda functions
    cdef np.ndarray[np.float64_t, ndim = 1] layer_outputs
    for layer, weight in enumerate(weights):
        layer_outputs = np.dot(neurons[layer], weight)
        neurons[layer+1] = activation_list[layer](layer_outputs)
    return neurons[-1]






