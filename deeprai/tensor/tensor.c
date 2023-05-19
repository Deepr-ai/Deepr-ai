#include <Python.h>
#include <stdlib.h>

typedef struct {
    double* data;
    int size;
} Tensor;

Tensor* create_tensor(int size) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = (double*)malloc(size * sizeof(double));
    tensor->size = size;
    return tensor;
}

void fill_tensor(Tensor* tensor, double* values) {
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = values[i];
    }
}

void print_tensor(Tensor* tensor) {
    printf("Tensor [");
    for (int i = 0; i < tensor->size; i++) {
        printf("%f", tensor->data[i]);
        if (i < tensor->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void destroy_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor);
}

// Define the tensor module
static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT,
    "tensor",
    "A module for tensor operations",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

// Define the PyInit_tensor function
PyMODINIT_FUNC PyInit_tensor(void) {
    return PyModule_Create(&tensor_module);
}
