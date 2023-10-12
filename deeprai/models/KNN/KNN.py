from deeprai.engine.build_model import Build
from deeprai.engine.base_layer import DistanceIndex
from deeprai.engine.cython.knn import knn
import numpy as np

class KNN():
    def __init__(self):
        self.builder = Build()
        self.x_vals = None
        self.y_vales = None
        self.p = None
        self.k = None

    def config_distance(self, distance):
        self.builder.translate_distance(distance)

    def store_vals(self, x_values, y_vales, p=3, k=2):
        self.x_vals = x_values
        self.y_vales = y_vales.astype(np.int32)
        self.p = p
        self.k = k

    def classify(self, query_point):
        return knn(X_train=self.x_vals, y_train=self.y_vales, query_point=query_point, distance_metric=DistanceIndex[0], p=self.p, k=self.k)


    def instant_classifier(self, x_vals, y_vals, query_point, p=3, k=2):
        y_vals = y_vals.astype(np.int32)
        return knn(x_vals, y_vals, query_point, distance_metric=int(DistanceIndex[0]), p=p, k=k)

    def classify_probability(self, query_point, expected_val):
        neighbors = self.classify_neighbors(query_point)
        positive_neighbors = sum([expected_val for index in neighbors if self.y_vales[index] == expected_val])
        probability = (positive_neighbors / len(neighbors)) * 100
        return probability

    def classify_neighbors(self, query_point):
        return knn(X_train=self.x_vals, y_train=self.y_vales, query_point=query_point, distance_metric=DistanceIndex[0],
                   p=self.p, k=self.k, return_neighbors=True)

    # for auto complete
    def hamming_distance(self):
        return "hamming distance"

    def minkowski_distance(self):
        return "minkowski distance"

    def manhattan_distance(self):
        return "manhattan distance"

    def euclidean_distance(self):
        return "euclidean distance"
