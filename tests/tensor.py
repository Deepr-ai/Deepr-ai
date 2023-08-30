from deeprai.tensor.tensor import Tensor
a = [[[0.0 for k in range(3)] for j in range(3)] for i in range(3)]
x = Tensor(a)
print(x.ndim)
print(x.shape)
print(x.to_list())
