import numpy as np
import random
import deeprai.models as model
inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(3000)])
expected = np.array([[i[0] + i[1]] for i in inputs])
net = model.FeedForward()
net.add_dense(2)
net.add_dense(5)
net.add_dense(1)
net.train_model(input_data=inputs, verify_data=expected, batch_size=36, epochs=100)
net.run(np.array([.5,.6]))

