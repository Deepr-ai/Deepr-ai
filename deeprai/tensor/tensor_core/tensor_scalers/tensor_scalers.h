PyObject* Tensor_add_scalar(TensorObject *self, PyObject *args);
PyObject* py_Tensor_update_with_scalar(TensorObject *self, PyObject *args);
float *operate_scalar(TensorObject *tensor, float scalar, char operation);
