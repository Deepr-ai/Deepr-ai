from deeprai.models import KNN
import numpy as np

model = KNN()
np.random.seed(42)
X_train_large = np.random.rand(100, 2) * 10
y_train_large = np.array([1 if np.linalg.norm(point - np.array([5, 5])) < 3 else 0 for point in X_train_large], dtype=np.int32)

query_point_large = np.array([5, 5], dtype=np.float64)

a = model.instant_classifier(X_train_large, y_train_large, query_point_large, k=10)
print(a)