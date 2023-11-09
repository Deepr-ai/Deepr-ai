from deeprai.models import Convolutional
import numpy as np
from Datasets.MNIST import mnist as db
# Loading in inputs
inputs = db.load_x(60000)
expected = db.load_y(60000)

# Loading in tests

test_x = db.load_x(10000)
test_y = db.load_y(10000)

inputs = inputs.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)


CNN = Convolutional()
CNN.input_shape((28,28))
CNN.add_conv(filters=2, kernel_size=(4, 4),stride=1, padding=0, actavation=CNN.relu)
CNN.add_pool((3, 3), "avr")
CNN.flatten()
CNN.add_dense(34)
CNN.add_dense(10)
CNN.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=1, epochs=3)
