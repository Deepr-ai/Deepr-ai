from deeprai import models
from deeprai.tools.toolkit import round_out
from Datasets.MNIST import mnist as db

inputs = db.load_x(60000)
expected = db.load_y(60000)

test_x = db.load_x(10000)
test_y = db.load_y(10000)

network = models.FeedForward()
network.config(loss="categorical cross entropy")
network.add_dense(784)
network.add_dense(60, activation='tanh')
network.add_dense(10, activation='sigmoid')

network.train_model(train_inputs=inputs, train_targets=expected, test_inputs=test_x, test_targets=test_y,
                    batch_size=136,
                    epochs=3)
print(expected[8081])
print(round_out(network.run(inputs[8081]), 3))
