import numpy as np
import random
import deeprai.models as model

inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(3000)])
expected = np.array([[i[0] + i[1]] for i in inputs])

network = model.FeedForward()

network.add_dense(2)

network.add_dense(5, activation='linear')

network.add_dense(1, activation='linear')

network.train_model(input_data=inputs, verify_data=expected, epochs=20)

output = network.run(np.array([.3,.1]))
print(output)