from . import tensor as _tensor_module
# from tensor_core.tensor_scalers import tensor_scaler



class Tensor:
    def __init__(self, data):
        if not self._validate_nd_tensor(data):
            raise ValueError("Input should be a list or tuple of lists/tuples representing an nD tensor")

        # Convert the python list to the C Tensor object
        self._tensor = _tensor_module.Tensor(data)

    def __getitem__(self, index):
        # Convert Python single integers or slices to tuple
        if not isinstance(index, tuple):
            index = (index,)

        return self._tensor[index]

    def __setitem__(self, index, value):
        # Convert Python single integers or slices to tuple
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) != self._tensor.ndim:
            raise IndexError("Number of indices do not match tensor dimensions")

        # Validate the value type here.
        if not isinstance(value, (int, float)):
            raise ValueError("Tensor values must be floats or integers")

        # Set the value using the underlying C Tensor object
        self._tensor[index] = value


    @property
    def shape(self):
        return self._tensor.shape

    @property
    def ndim(self):
        return len(self.shape)

    def _validate_nd_tensor(self, data):
        """Recursive function to validate n-dimensional tensor."""
        if not isinstance(data, (list, tuple)):
            return False

        if not data:  # empty list or tuple
            return True

        first_elem = data[0]
        if isinstance(first_elem, (list, tuple)):
            depth = len(data[0])
            return all(len(item) == depth and self._validate_nd_tensor(item) for item in data)
        else:
            return all(isinstance(item, (int, float)) for item in data)

