import numpy as np
import random
import deeprai.models as model

inputs = np.array([[round(random.uniform(600, 1000), 2)] for _ in range(4000)])
expected = np.array([[np.log2(i[0])] for i in inputs])

test_inputs = np.array([[round(random.uniform(600, 1000), 2)] for _ in range(500)])
test_expected = np.array([[np.log2(i[0])] for i in inputs])

inputs = np.divide(inputs,100)
expected = np.divide(expected,100)
test_inputs = np.divide(test_inputs,100)
test_expected = np.divide(test_expected,100)


network = model.FeedForward()
network.config(loss="categorical cross entropy")
network.add_dense(1)
network.add_dense(90, dropout=0.001)
network.add_dense(1, activation='linear')

network.train_model(train_inputs=inputs,
                    train_targets=expected,
                    test_inputs=test_inputs,
                    test_targets=test_expected,
                    epochs=60)

network.graph(metric="cost")
network.specs()

