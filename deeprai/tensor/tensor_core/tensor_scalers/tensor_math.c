#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../libs/tensor_shared.h"

// In-place update function
void Tensor_update_with_scalar(TensorObject *tensor, float scalar, char operation) {
    for (int i = 0; i < tensor->size; i++) {
        switch (operation) {
            case '+': tensor->data[i] += scalar; break;
            case '-': tensor->data[i] -= scalar; break;
            case '*': tensor->data[i] *= scalar; break;
            case '/':
                if (scalar == 0.0f) {
                    PyErr_SetString(PyExc_ZeroDivisionError, "Division by zero.");
                    return;
                }
                tensor->data[i] /= scalar;
                break;
            case '%':
                if (scalar == 0.0f) {
                    PyErr_SetString(PyExc_ZeroDivisionError, "Modulo by zero.");
                    return;
                }
                tensor->data[i] = fmod(tensor->data[i], scalar);
                break;
            // ... Add more operations
        }
    }
}

// Function to create a new tensor with result data after operation
float *operate_scalar(TensorObject *tensor, float scalar, char operation) {
    float *result = malloc(tensor->size * sizeof(float));
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for tensor data.");
        return NULL;
    }

    memcpy(result, tensor->data, tensor->size * sizeof(float));

    TensorObject tempTensor;
    tempTensor.data = result;
    tempTensor.size = tensor->size;
    Tensor_update_with_scalar(&tempTensor, scalar, operation);

    if (PyErr_Occurred()) {
        free(result);
        return NULL;
    }

    return result;
}

static PyObject* Tensor_add_scalar(TensorObject *self, PyObject *args) {
    float scalar;
    if (!PyArg_ParseTuple(args, "f", &scalar))
        return NULL;

    float *result_data = operate_scalar(self, scalar, '+');
    if (!result_data) {
        return NULL;  // Error has been set by operate_scalar
    }

    // Create a new TensorObject with result_data
    TensorObject *result_tensor = PyObject_New(TensorObject, &TensorType);  // Assuming TensorType is defined elsewhere
    result_tensor->data = result_data;
    result_tensor->shape = self->shape;  // Just pointing to the shape, assuming tensor shape doesn't change.
    result_tensor->ndim = self->ndim;
    result_tensor->size = self->size;

    return (PyObject *)result_tensor;
}

static PyObject* py_Tensor_update_with_scalar(TensorObject *self, PyObject *args) {
    float scalar;
    char operation;
    if (!PyArg_ParseTuple(args, "fc", &scalar, &operation)) {
        return NULL;
    }

    Tensor_update_with_scalar(self, scalar, operation);

    if (PyErr_Occurred()) {
        return NULL;
    }

    Py_RETURN_NONE;
}
static PyMethodDef TensorScaler_methods[] = {
    {"add_scalar", (PyCFunction)Tensor_add_scalar, METH_VARARGS, "Add a scalar to the tensor"},
    {"update_with_scalar", (PyCFunction)py_Tensor_update_with_scalar, METH_VARARGS, "Update tensor with scalar operation"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject TensorScalerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensor.tensor_scaler",
    .tp_doc = "Tensor scaler objects",
    .tp_methods = TensorScaler_methods,
};

static PyModuleDef tensorScalersModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tensor_scaler",
    .m_doc = "Tensor scalers module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_tensor_scaler(void) {
    PyObject *m;

    // If you have a new Python type in tensor_math.c
    if (PyType_Ready(&TensorScalerType) < 0)
        return NULL;

    m = PyModule_Create(&tensorScalersModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorScalerType);
    PyModule_AddObject(m, "tensor_scaler", (PyObject *)&TensorScalerType);

    return m;
};
