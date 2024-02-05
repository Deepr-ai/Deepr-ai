// tensor_shared.h

#ifndef TENSOR_SHARED_H
#define TENSOR_SHARED_H

#include <Python.h>

typedef struct {
    PyObject_HEAD
    float *data;
    int *shape;
    int ndim;
    int size;
} TensorObject;
extern PyTypeObject TensorType;


#endif  // TENSOR_SHARED_H
