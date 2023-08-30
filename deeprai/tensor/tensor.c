#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    int *data;
    int size;
} TensorObject;

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    TensorObject *self;
    self = (TensorObject *) type->tp_alloc(type, 0);
    if (self) {
        self->data = NULL;
        self->size = 0;
    }
    return (PyObject *) self;
}


static int Tensor_init(TensorObject *self, PyObject *args, PyObject *kwds) {
    PyObject *dataList;
    if (!PyArg_ParseTuple(args, "O", &dataList))
        return -1;

    if (!PyList_Check(dataList))
        return -1;

    self->size = PyList_Size(dataList);
    self->data = (int*)malloc(self->size * sizeof(int));

    for (int i = 0; i < self->size; i++) {
        PyObject *value = PyList_GetItem(dataList, i);
        self->data[i] = PyLong_AsLong(value);
    }


    return 0;
}

static void Tensor_dealloc(TensorObject *self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* Tensor_getitem(TensorObject *self, PyObject *key) {
    if (!PyLong_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Tensor indices must be integers");
        return NULL;
    }
    Py_ssize_t i = PyLong_AsSsize_t(key);
    if (i < 0) i += self->size;
    if (i < 0 || i >= self->size) {
        PyErr_SetString(PyExc_IndexError, "Tensor index out of range");
        return NULL;
    }
    return PyLong_FromLong(self->data[i]);
}



static PyMappingMethods Tensor_as_mapping = {
    .mp_length = NULL,        // Implement if you want the "len" function to work
    .mp_subscript = (binaryfunc)Tensor_getitem,
    .mp_ass_subscript = NULL, // Implement if you want to support item assignment
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensor.tensor_core.Tensor",
    .tp_doc = "Tensor objects",
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Tensor_new,
    .tp_init = (initproc) Tensor_init,
    .tp_dealloc = (destructor) Tensor_dealloc,
    .tp_as_mapping = &Tensor_as_mapping,
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
