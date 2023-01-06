import deeprai.models.convolutional as base
import numpy as np
class StackEvents:
    def __init__(self):
        self.NetworkQueue = []
    def create_kernel(self, amount, shape, max_size):
        self.NetworkQueue.append("kernel")
        local_kernels = []
        for val in range(amount):
            kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
            local_kernels.append(kernel.tolist())
        base.CNN.
            append(local_kernels)
        return local_kernels
    def create_pool(self):
        pass

    def create_dense(self):
        pass
    def create_flat(self):
        pass

    def create_merge(self):
        pass