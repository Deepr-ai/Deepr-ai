import deeprai
a = [[[0.0 for k in range(3)] for j in range(3)] for i in range(3)]
x = deeprai.Tensor(a)
print(x.to_list())
