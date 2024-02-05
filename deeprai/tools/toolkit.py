import numpy as np


def verify_inputs(array):
    """
    Verify if the input is a numpy array.

    Args:
        array: Input to be checked.

    Returns:
        bool: True if input is a numpy array, False otherwise.
    """
    return isinstance(array, np.ndarray)


def round_out(array, a=2):
    """
    Round the elements of the numpy array and set print options.

    Args:
        array (np.ndarray): The input numpy array.
        a (int, optional): Decimal places to round to. Defaults to 2.

    Returns:
        np.ndarray: Rounded numpy array.
    """
    np.set_printoptions(precision=a, suppress=True, floatmode='fixed')
    return np.round(array, a)


def normalize(array):
    """
    Normalize the numpy array to the range [0, 1].

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


def reshape_to_2d(array):
    """
    Reshape the numpy array to 2D if it's not already.

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray: Reshaped 2D array.
    """
    return array.reshape(-1, array.shape[-1])


def is_square_matrix(array):
    """
    Check if a numpy array is a square matrix.

    Args:
        array (np.ndarray): Input array.

    Returns:
        bool: True if array is a square matrix, False otherwise.
    """
    return array.ndim == 2 and array.shape[0] == array.shape[1]


def sum_along_axis(array, axis=0):
    """
    Compute the sum along a specified axis.

    Args:
        array (np.ndarray): Input array.
        axis (int, optional): Axis along which the sum is computed. Defaults to 0.

    Returns:
        np.ndarray: Sum along the specified axis.
    """
    return np.sum(array, axis=axis)


def array_info(array):
    """
    Get information about the numpy array.

    Args:
        array (np.ndarray): Input array.

    Returns:
        dict: A dictionary containing shape, dtype, min and max values.
    """
    info = {
        'shape': array.shape,
        'dtype': array.dtype,
        'min_value': np.min(array),
        'max_value': np.max(array)
    }
    return info
