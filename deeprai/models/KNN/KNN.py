from deeprai.engine.build_model import Build
from deeprai.engine.base_layer import DistanceIndex
from deeprai.engine.cython.knn import knn
class KNN(Build):
    def __init__(self):
        super().__init__()

    def config_distance(self, distance):
        self.translate_distance(distance)

    def train(self, x_vals, y_vals, query_point, p=3, k=0):
        return knn(x_vals, y_vals, query_point, distance_metric=DistanceIndex, p=p, k=k)

    # for auto complete
    def hamming_distance(self):
        return "hamming distance"

    def minkowski_distance(self):
        return "minkowski distance"

    def manhattan_distance(self):
        return "manhattan distance"

    def euclidean_distance(self):
        return "euclidean distance"