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
        self.y_vales = y_vales
        self.p = p
        self.k = k

    def classify(self, query_point):
        return knn(self.x_vals, self.y_vales, query_point, DistanceIndex[0], self.p, self.k,)

    def instant_classifier(self, x_vals, y_vals, query_point, p=3, k=2):
        y_vals = y_vals.astype(np.int32)  # Ensure y_vals is of type int32
        return knn(x_vals, y_vals, query_point, distance_metric=int(DistanceIndex[0]), p=p, k=k)

    # for auto complete
    def hamming_distance(self):
        return "hamming distance"

    def minkowski_distance(self):
        return "minkowski distance"

    def manhattan_distance(self):
        return "manhattan distance"

    def euclidean_distance(self):
        return "euclidean distance"
