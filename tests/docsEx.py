import numpy as np
import random
import deeprai.models as model

inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(3000)])
expected = np.array([[i[0] + i[1]] for i in inputs])

network = model.FeedForward()

network.add_dense(2)

network.add_dense(50, dropout=0.1)

network.add_dense(1)

network.train_model(train_inputs=inputs, train_targets=expected, test_inputs=inputs, test_targets=expected, epochs=100)

network.graph(metric="acc")
network.graph()