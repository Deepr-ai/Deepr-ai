from . import tensor as _tensor_module


class Tensor:
    def __init__(self, data):
        if not isinstance(data, list):
            raise ValueError("Input should be a list")

        # Convert the python list to the C Tensor object
        self._tensor = _tensor_module.Tensor(data)

    def __getitem__(self, index):
        return self._tensor[index]