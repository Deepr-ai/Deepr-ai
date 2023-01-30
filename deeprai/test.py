import numpy as np
import random
from deeprai import models
inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] + i[1]] for i in inputs])

models = models.FeedForward()
#add input layer
models.add_dense(2, activation='sigmoid')
#add hidden layer
models.add_dense(11, activation='sigmoid')
#add output layer
models.add_dense(1, activation='sigmoid')
#optinal but default is optimizer is gradient decent, default loss is mean square error
models.model_optimizers(optimizer='gradient decent', loss='categorical cross entropy')
# trains the model
models.train_model(input_data=inputs, verify_data=targets, batch_size=36, epochs=500)
