from deeprai import models
# Importing deeprai-datasets lib, only available on linux
from Datasets.MNIST import mnist as db
from deeprai.tools.noise import GaussianNoise
X = []
# Loading in inputs
inputs = db.load_x(5000)
expected = db.load_y(5000)
y = GaussianNoise()

# Loading in tests
test_x = db.load_x(500)
test_y = db.load_y(500)

# Spawning the model
network = models.FeedForward()

# Creating dense layers
network.add_dense(784)
network.add_dense(60, activation=network.relu)
network.add_dense(10, activation=network.sigmoid)
network.config(loss=network.mean_square_error, optimizer=network.adam)
network.checkpoint(interval=2, dir_location="checkpoints")
# Training the model
network.train_model(train_inputs=inputs, train_targets=expected,
                    test_inputs=test_x, test_targets=test_y,
                    batch_size=130, epochs=10)

# Evaluating the model
stats = network.evaluate(test_x, test_y)
print(f"Cost: {stats['cost']}")
print(f"Accuracy: {stats['accuracy']}")

# Saving the model
# network.save("MNIST1")