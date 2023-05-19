import ctypes

# Load the C library containing the tensor operations
tensor_functions = ctypes.cdll.LoadLibrary('tensor.so')

# Define the tensor class
class tensor:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))
        self.num_dims = len(self.shape)
        self.data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Function to create a new tensor with the same shape as this tensor
    def zeros_like(self):
        new_data = [[0 for j in range(self.shape[1])] for i in range(self.shape[0])]
        return tensor(new_data)

    # Function to get the value of the tensor at the given index
    def __getitem__(self, indices):
        indices = (ctypes.c_int * len(indices))(*indices)
        value = ctypes.c_float(tensor_functions.get_value(self.data_ptr, self.shape, self.num_dims, indices))
        return value.value

    # Function to set the value of the tensor at the given index
    def __setitem__(self, indices, value):
        indices = (ctypes.c_int * len(indices))(*indices)
        value = ctypes.c_float(value)
        tensor_functions.set_value(self.data_ptr, self.shape, self.num_dims, indices, value)

    # Function to add two tensors element-wise
    def __add__(self, other):
        assert self.shape == other.shape, "Shapes of tensors must match"
        new_data = [[0 for j in range(self.shape[1])] for i in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                new_data[i][j] = self.data[i][j] + other.data[i][j]
        return tensor(new_data)
