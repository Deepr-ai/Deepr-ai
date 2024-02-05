from deeprai.models import FeedForward
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib as pbl

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Converting to np array
Data_X = np.array(X).astype(np.float64)
Data_Y = np.where(np.array(y) == 'M', [0, 1], [1, 0]).astype(np.float64)

# Model ~ 92% acc
model = FeedForward()
model.add_dense(30)
model.add_dense(50, activation=model.tanh)
model.add_dense(2, activation=model.sigmoid)
model.config(optimizer=model.rmsprop)
model.train_model(Data_X, Data_Y, Data_X, Data_Y, 1, 100)