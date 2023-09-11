import numpy as np
import random
import deeprai.models as model

# Creating inputs/tests
inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(3000)])
expected = np.array([[i[0] + i[1]] for i in inputs])

# Spawning in the model
network = model.FeedForward()

# Creating the dense layers
network.add_dense(2)

network.add_dense(5, activation='linear')

network.add_dense(1, activation='sigmoid')

# Training the model
network.train_model(train_inputs=inputs,train_targets=expected,test_inputs=inputs,test_targets=expected,epochs=20)

# Running an input through the network
output = network.run(np.array([.3,.1]))
print(output)