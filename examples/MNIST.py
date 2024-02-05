from deeprai import models
from Datasets.MNIST import mnist as db
# Loading in inputs
inputs = db.load_x(10000)
expected = db.load_y(10000)
# Loading in tests
test_x = db.load_x(1000)
test_y = db.load_y(1000)
network = models.FeedForward()

# Creating dense layers
network.add_dense(784)
network.add_dense(70, activation=network.relu)
network.add_dense(10, activation=network.softmax)
network.config(loss=network.cross_entropy, optimizer=network.adam)
# Training the model
network.specs()
network.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=36, epochs=100)