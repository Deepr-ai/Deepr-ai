from deeprai.models import KNN
import numpy as np

class CollegeAdmissionPredictor:
    def __init__(self):
        self.knn = KNN()
        self.data = np.array([
            [1340, 3.8, 1],
            [1300, 3.5, 0],
            [1510, 4.0, 1],
            [1250, 3.2, 0],
            [1450, 3.9, 1],
            [1350, 3.6, 1],
            [1380, 3.7, 1],
            [1280, 3.3, 0],
            [1420, 3.8, 1],
            [1320, 3.4, 1],
            [1480, 3.9, 1],
            [1260, 3.1, 0],
            [1430, 3.7, 1],
            [1290, 3.4, 0],
            [1370, 3.5, 1],
            [1360, 3.4, 1],
            [1390, 3.6, 1],
            [1400, 3.5, 1],
            [1410, 3.7, 1],
            [1440, 3.8, 1],
            [1460, 3.9, 1],
            [1470, 4.0, 1],
            [1490, 3.9, 1],
            [1500, 4.0, 1],
            [1330, 3.3, 0],
            [1310, 3.2, 0]
        ])

        self.x_vals = self.data[:, :-1]
        self.y_vals = self.data[:, -1]
        self.knn.store_vals(self.x_vals, self.y_vals, k=8)

    def predict_admission(self, SAT_score, GPA):
        query_point = np.array([SAT_score, GPA])
        return self.knn.classify(query_point)

    def predict_admission_probability(self, SAT_score, GPA):
        query_point = np.array([SAT_score, GPA])
        return self.knn.classify_probability(query_point)

    def show_k_nearest_neighbors(self, SAT_score, GPA):
        query_point = np.array([SAT_score, GPA])
        return self.knn.classify_neighbors(query_point)


predictor = CollegeAdmissionPredictor()
SAT_score = float(input("Enter your SAT score: "))
GPA = float(input("Enter your GPA: "))

admission_result = predictor.predict_admission(SAT_score, GPA)
admission_probability = predictor.predict_admission_probability(SAT_score, GPA)

print(f"You are {admission_probability:.2f}% likely to get admitted.")
