import numpy as np
cimport numpy as np
from knn_distance import *


cpdef knn(np.ndarray[np.float64_t, ndim=2] X_train, np.ndarray[int, ndim=1] y_train,
        np.ndarray[np.float64_t, ndim=1] query_point, int k=3, distance_metric=0, int p=3):
    cdef np.ndarray[np.float64_t, ndim=1] distances
    cdef np.ndarray[int, ndim=1] sorted_indices
    cdef np.ndarray[int, ndim=1] k_nearest_labels

    distance_functions = [euclidean_distance, manhattan_distance, minkowski_distance, hamming_distance]

    # Calculate distances using the selected distance metric
    if distance_metric == 2:  # Minkowski requires an additional 'p' parameter
        distances = distance_functions[distance_metric](X_train, query_point, p)
    else:
        distances = distance_functions[distance_metric](X_train, query_point)

    sorted_indices = np.argsort(distances)
    k_nearest_labels = y_train[sorted_indices[:k]]

    # Majority voting
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]