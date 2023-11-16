from deeprai.models import Convolutional
import numpy as np
from Datasets.MNIST import mnist as db
# Loading in inputs
inputs = db.load_x(2000)
expected = db.load_y(2000)

# Loading in tests

test_x = db.load_x(500)
test_y = db.load_y(500)

inputs = inputs.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)


CNN = Convolutional()
CNN.input_shape((28,28,1))
CNN.add_conv(filters=32, kernel_size=(3, 3),stride=1, padding=0, actavation=CNN.relu)
CNN.add_pool((2,2), "avr")
CNN.flatten()
CNN.add_dense(100, activation=CNN.relu)
CNN.add_dense(10)
CNN.config(optimizer=CNN.adam)
CNN.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=10, epochs=5, learning_rate=.4)

print(CNN.evaluate(test_x, test_y))

