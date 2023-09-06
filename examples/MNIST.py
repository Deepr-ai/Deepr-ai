from deeprai import models

# Importing deeprai-datasets lib, only available on linux
from Datasets.MNIST import mnist as db
from deeprai.tools.noise import SpeckleNoise
# Loading in inputs
inputs = db.load_x(60000)
expected = db.load_y(60000)
y = SpeckleNoise()

# Loading in tests
test_x = db.load_x(10000)
test_y = db.load_y(10000)

# Spawning the model
network = models.FeedForward()
network.config(loss="categorical cross entropy")

# Creating dense layers
network.add_dense(784)
network.add_dense(60, activation='tanh', dropout=.02)
network.add_dense(10, activation='sigmoid')

# Training the model
network.train_model(train_inputs=y.noise(arrays=inputs), train_targets=expected, test_inputs=y.noise(arrays=test_x), test_targets=test_y,
                    batch_size=130, epochs=2)

# Saving the model
