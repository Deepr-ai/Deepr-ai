from deeprai import models
from deeprai.activation import softmax
from Datasets.MNIST import mnist as db
s = softmax.Softmax()


inputs = db.load_x(60000)
expected = db.load_y(60000)

test_x = db.load_x(10000)
test_y = db.load_y(10000)

network = models.FeedForward()
network.add_dense(784)
network.add_dense(60, activation='tanh')
network.add_dense(10, activation='sigmoid')
network.config(loss='categorical cross entropy')
network.specs()
network.train_model(input_data=inputs, verify_data=expected, test_input=test_x,test_targets=test_y, batch_size=136,
                    epochs=200, early_stop=True)
network.save("MNIST.deepr")

print(network.run(test_x[666]))
print(s.softmax(network.run(test_x[666])))
print(test_y[666])