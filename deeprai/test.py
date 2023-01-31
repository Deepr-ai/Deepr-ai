import numpy as np
import random
from deeprai import models
from keras.datasets import mnist
# inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(3000)])
# expected = np.array([[i[0] + i[1]] for i in inputs])
(train_X, train_y), (test_X, test_y) = mnist.load_data()
x = []
y = []
for i in range(10000):
    x.append([item for sublist in train_X[i] for item in sublist])
    y1 = [0,0,0,0,0,0,0,0,0,0,0]
    y1[train_y[i]] = 1.0
    y.append(y1)
inputs = np.array(x)
expected = np.array(y)

inputs = inputs.astype(np.float64)
expected = expected.astype(np.float64)

models = models.FeedForward()
#add input layer
models.add_dense(784, activation='relu')
#add hidden layer
models.add_dense(20, activation='sigmoid')
#add output layer
models.add_dense(11, activation='sigmoid')
#optinal but default is optimizer is gradient decent, default loss is mean square error
models.model_optimizers(optimizer='gradient decent', loss='mean square error')
# trains the model
models.train_model(input_data=inputs, verify_data=expected, batch_size=36, epochs=100)
# models.run(np.array([0.4,.01]))
# models.save("sum3.deepr")
print(3)