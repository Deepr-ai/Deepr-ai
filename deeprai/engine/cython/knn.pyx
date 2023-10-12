import numpy as np
cimport numpy as np
from deeprai.engine.cython.knn_distance import euclidean_distance, manhattan_distance, minkowski_distance, hamming_distance

def knn(np.ndarray[np.float64_t, ndim=2] X_train, np.ndarray[np.int32_t, ndim=1] y_train,
        np.ndarray[np.float64_t, ndim=1] query_point, int k=3, int distance_metric=0, int p=3, bint return_neighbors=False):
    cdef np.ndarray[np.float64_t, ndim=1] distances
    cdef np.ndarray[np.int64_t, ndim=1] sorted_indices
    cdef np.ndarray[np.int32_t, ndim=1] k_nearest_labels

    # Ensure k has a valid value
    if k <= 0 or k > X_train.shape[0]:
        raise ValueError(f"Invalid value of k: {k}. It should be between 1 and {X_train.shape[0]}.")

    distance_functions = [euclidean_distance, manhattan_distance, minkowski_distance, hamming_distance]
    distances = np.zeros(X_train.shape[0], dtype=np.float64)
    for i in range(X_train.shape[0]):
        if distance_metric == 2:  # Minkowski requires an additional 'p' parameter
            distances[i] = distance_functions[distance_metric](X_train[i], query_point, p)
        else:
            distances[i] = distance_functions[distance_metric](X_train[i], query_point)

    sorted_indices = np.argsort(distances)
    k_nearest_indices = sorted_indices[:k]
    k_nearest_labels = y_train[k_nearest_indices]

    if return_neighbors:
        return k_nearest_indices
    else:
        # Majority voting
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
