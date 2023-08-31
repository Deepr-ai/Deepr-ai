#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "libs/tensor_shared.h"




static int get_dimensions(PyObject *list) {
    int dims = 0;
    while (PyList_Check(list)) {
        list = PyList_GetItem(list, 0);
        dims++;
    }
    return dims;
}

static int get_size_at_dim(PyObject *list, int dim) {
    for (int i = 0; i < dim; i++) {
        if (PyList_Check(list)) {
            list = PyList_GetItem(list, 0);
        } else {
            return -1;
        }
    }
    return PyList_Size(list);
}

static void fill_data_recursive(TensorObject *self, PyObject *list, int dim, int *index) {
    int size = PyList_Size(list);
    for (int i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(list, i);
        if (dim < self->ndim - 1) {
            fill_data_recursive(self, item, dim + 1, index);
        } else {
            self->data[*index] = (float)PyFloat_AsDouble(item);
            (*index)++;
        }
    }
}

static int Tensor_init(TensorObject *self, PyObject *args, PyObject *kwds) {
    PyObject *dataList;
    if (!PyArg_ParseTuple(args, "O", &dataList))
        return -1;

    if (!PyList_Check(dataList))
        return -1;

    self->ndim = get_dimensions(dataList);
    self->shape = (int *)malloc(self->ndim * sizeof(int));

    int totalSize = 1;
    for (int i = 0; i < self->ndim; i++) {
        int dimSize = get_size_at_dim(dataList, i);
        if (dimSize == -1) return -1;

        self->shape[i] = dimSize;
        totalSize *= dimSize;
    }
    self->size = totalSize;

    self->data = (float *)malloc(self->size * sizeof(float));

    int startIndex = 0;
    fill_data_recursive(self, dataList, 0, &startIndex);

    return 0;
}

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    TensorObject *self;

    // Allocate memory for the new Tensor object
    self = (TensorObject *) type->tp_alloc(type, 0);

    // Check if memory allocation was successful
    if (!self) {
        return NULL;
    }

    // Initialize the data and size to default values
    self->data = NULL;
    self->size = 0;

    // Return the newly allocated object
    return (PyObject *) self;
}


static void Tensor_dealloc(TensorObject *self) {
    free(self->data);
    free(self->shape);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* Tensor_getitem(TensorObject *self, PyObject *key) {
    // For simplicity, only tuple-based indexing is supported
    if (!PyTuple_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Multi-dimensional Tensor indices must be tuples");
        return NULL;
    }

    Py_ssize_t index = 0;  // change int to Py_ssize_t
    Py_ssize_t multiplier = 1;  // change int to Py_ssize_t
    for (int i = self->ndim - 1; i >= 0; i--) {
        PyObject *dimIndexObj = PyTuple_GetItem(key, i);
        if (!PyLong_Check(dimIndexObj)) {
            PyErr_SetString(PyExc_TypeError, "Each index in the tuple must be an integer");
            return NULL;
        }

        int dimIndex = (int)PyLong_AsLong(dimIndexObj);
        if (dimIndex < 0 || dimIndex >= self->shape[i]) {
            PyErr_SetString(PyExc_IndexError, "Tensor index out of range");
            return NULL;
        }

        index += dimIndex * multiplier;
        multiplier *= self->shape[i];
    }

    return PyFloat_FromDouble(self->data[index]);
}

// Set values in the tensor
static int Tensor_setitem(TensorObject *self, PyObject *key, PyObject *value) {
    if (!PyTuple_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Multi-dimensional Tensor indices must be tuples");
        return -1;
    }

    if (!PyFloat_Check(value) && !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Tensor values must be floats or integers");
        return -1;
    }

    int index = 0;
    int multiplier = 1;
    for (int i = self->ndim - 1; i >= 0; i--) {
        PyObject *dimIndexObj = PyTuple_GetItem(key, i);
        if (!PyLong_Check(dimIndexObj)) {
            PyErr_SetString(PyExc_TypeError, "Each index in the tuple must be an integer");
            return -1;
        }

        int dimIndex = (int)PyLong_AsLong(dimIndexObj);
        if (dimIndex < 0 || dimIndex >= self->shape[i]) {
            PyErr_SetString(PyExc_IndexError, "Tensor index out of range");
            return -1;
        }

        index += dimIndex * multiplier;
        multiplier *= self->shape[i];
    }

    self->data[index] = (float)PyFloat_AsDouble(value);
    return 0;  // success
}

// Recursive function to build nested lists
static PyObject* tensor_to_list_recursive(TensorObject *self, int *indices, int depth) {
    if (depth == self->ndim) {
        int flat_index = 0;
        int multiplier = 1;
        for (int i = 0; i < self->ndim; i++) {
            flat_index += indices[i] * multiplier;
            multiplier *= self->shape[i];
        }
        return PyFloat_FromDouble(self->data[flat_index]);
    }

    PyObject *list = PyList_New(self->shape[depth]);

    for (int i = 0; i < self->shape[depth]; i++) {
        indices[depth] = i;
        PyObject *sub_list = tensor_to_list_recursive(self, indices, depth + 1);
        if (!sub_list) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, sub_list);  // No need to DECREF sub_list, PyList_SET_ITEM steals a reference
    }

    return list;
}

static PyObject* Tensor_to_list(TensorObject *self, PyObject *Py_UNUSED(ignored)) {
    int *indices = (int *)calloc(self->ndim, sizeof(int));
    if (!indices) {
        PyErr_NoMemory();
        return NULL;
    }
    PyObject *list = tensor_to_list_recursive(self, indices, 0);
    free(indices);
    return list;
}

static PyObject* Tensor_get_ndim(TensorObject *self, void *closure) {
    return PyLong_FromLong(self->ndim);
}


static PyObject* Tensor_get_shape(TensorObject *self, void *closure) {
    PyObject *shapeTuple = PyTuple_New(self->ndim);
    if (!shapeTuple) return NULL;

    for (int i = 0; i < self->ndim; i++) {
        PyTuple_SetItem(shapeTuple, i, PyLong_FromLong(self->shape[i]));
    }

    return shapeTuple;
}

PyObject* Tensor_add_scalar(TensorObject *self, PyObject *args);
PyObject* py_Tensor_update_with_scalar(TensorObject *self, PyObject *args);

static PyMethodDef Tensor_methods[] = {
    {"to_list", (PyCFunction)Tensor_to_list, METH_NOARGS, "Convert the tensor to a nested Python list"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef Tensor_getseters[] = {
    {"shape", (getter)Tensor_get_shape, NULL, "shape of the tensor", NULL},
    {"ndim", (getter)Tensor_get_ndim, NULL, "number of dimensions of the tensor", NULL},
    {NULL}  /* Sentinel */
};



static PyMappingMethods Tensor_as_mapping = {
    .mp_length = NULL,
    .mp_subscript = (binaryfunc)Tensor_getitem,
    .mp_ass_subscript = (objobjargproc)Tensor_setitem,
};


 PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensor.Tensor",
    .tp_doc = "Tensor objects",
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Tensor_new,
    .tp_init = (initproc) Tensor_init,
    .tp_dealloc = (destructor) Tensor_dealloc,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_methods = Tensor_methods,
    .tp_getset = Tensor_getseters,
};

static PyModuleDef tensorModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tensor",
    .m_doc = "Tensor module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_tensor(void) {
    PyObject *m;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;

    m = PyModule_Create(&tensorModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorType);
    PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType);
    return m;
}
