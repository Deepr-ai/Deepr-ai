from deeprai import models
from Datasets.MNIST import mnist as db

inputs = db.load_x(60000)
expected = db.load_y(60000)

test_x = db.load_x(10000)
test_y = db.load_y(10000)

network = models.FeedForward()
network.add_dense(784)
network.add_dense(60, activation='tanh')
network.add_dense(10, activation='sigmoid')
network.specs()

network.train_model(input_data=inputs, verify_data=expected, test_input=test_x,test_targets=test_y, batch_size=136,
                    epochs=4)
network.graph(metric="acc")
network.clear_local_network()
print(network.run(inputs[4]))
print(expected[4])
