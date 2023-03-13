from deeprai import models
import numpy as np

# Linear test
# x = np.array([3., 5., 7., 9.])
# y = np.array([1., 2., 3., 4.])
#
# network = models.LinearRegression()
# print(network.fit(x, y))
# print(network.run(3))

# Poly test
x = np.arange(-24, 25, dtype=np.float64)
y = x**2
network = models.PolyRegression()
print(network.fit(x, y))