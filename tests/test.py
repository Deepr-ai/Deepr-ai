import numpy as np
from deeprai import models
from keras.datasets import mnist




(train_X, train_y), (test_X, test_y) = mnist.load_data()
x = []
y = []
for i in range(1000):
    x.append([item for sublist in train_X[i] for item in sublist])
    y1 = [0,0,0,0,0,0,0,0,0,0]
    y1[train_y[i]] = 1.0
    y.append(y1)

x1 = [[number / 1000 for number in sublist] for sublist in x]
inputs = np.array(x1)
expected = np.array(y)

inputs = inputs.astype(np.float64)
expected = expected.astype(np.float64)

network = models.FeedForward()

network.add_dense(784)

network.add_dense(50, activation='tanh')

network.add_dense(10, activation='sigmoid')

# trains the model
network.train_model(input_data=inputs, verify_data=expected, batch_size=36, epochs=100)