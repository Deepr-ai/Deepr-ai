from deeprai.models import Convolutional
import numpy as np
from Datasets.MNIST import mnist as db

# Loading in inputs
inputs = db.load_x(10000)
expected = db.load_y(10000)

# Loading in tests
test_x = db.load_x(1000)
test_y = db.load_y(1000)

inputs = inputs.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)

CNN = Convolutional()
CNN.input_shape((28,28,1))
# CNN.add_conv(filters=3, kernel_size=(3, 3),stride=1, actavation=CNN.linear)
# CNN.add_pool((3,3), "max")
CNN.flatten()
CNN.add_dense(70, activation=CNN.relu)
CNN.add_dense(10, activation=CNN.softmax)
CNN.config(optimizer=CNN.adam, loss=CNN.cross_entropy)
print(CNN.run(inputs[1]))
CNN.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=1, epochs=50)

# print(CNN.evaluate(inputs, expected))
# print("=================")
# print(CNN.run(inputs[1]))
# print(expected[1])
