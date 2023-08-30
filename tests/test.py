from deeprai import models
from Datasets.MNIST import mnist as db

inputs = db.load_x(60000)[10000:]
expected = db.load_y(60000)[10000:]

test_x = db.load_x(10000)
test_y = db.load_y(10000)

network = models.FeedForward()
network.config(loss="categorical cross entropy")
network.add_dense(784)
network.add_dense(60, activation='tanh', dropout=.02)
network.add_dense(10, activation='sigmoid')
network.specs()
network.train_model(train_inputs=inputs, train_targets=expected, test_inputs=test_x, test_targets=test_y,
                    batch_size=136)
network.save('MNIST.deepr')
