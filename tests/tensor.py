from deeprai.tools.noise import SpeckleNoise
import numpy as np


noise = SpeckleNoise()

arr = np.random.rand(3,2)
print(type(noise.noise(arr)))