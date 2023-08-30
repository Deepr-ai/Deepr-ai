from deeprai.tensor.tensor import Tensor

x = Tensor([1, 2, 3, 4, 5])  # prints 2
x[2] = 8
print(x[2])
