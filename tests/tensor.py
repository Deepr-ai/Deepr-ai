# import deeprai
# a = [[[0.0 for k in range(3)] for j in range(3)] for i in range(3)]
# x = deeprai.Tensor(a)
# print(x.to_list())

import importlib.util
spec = importlib.util.spec_from_file_location("tensor_scaler", "deeprai/tensor/tensor_scalers.cpython-311-x86_64-linux-gnu.so")
tensor_scaler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensor_scaler)

