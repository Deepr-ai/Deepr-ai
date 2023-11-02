from deeprai import models
# Importing deeprai-datasets lib, only available on linux
from Datasets.MNIST import mnist as db
# Loading in inputs
inputs = db.load_x(5000)
expected = db.load_y(5000)

# Loading in tests
test_x = db.load_x(1000)
test_y = db.load_y(1000)

# Spawning the model
network = models.FeedForward()

# Creating dense layers
network.add_dense(784)
network.add_dense(50, activation=network.relu)
network.add_dense(10, activation=network.sigmoid)
network.config(loss=network.cross_entropy, optimizer=network.adam)
# Training the model
network.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=130, epochs=10)




# Evaluating the model
stats = network.evaluate(inputs[:2], expected[:2])
print(f"Cost: {stats['cost']}")
print(f"Accuracy: {stats['accuracy']}")
#
# # Saving the model
# network.save("MNIST84k.deepr")