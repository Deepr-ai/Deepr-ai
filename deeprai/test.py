import numpy as np
import random
from deeprai import models
from deeprai.engine.base_layer import ActivationList, WeightVals, LayerVals
"""
Final syntax structure
"""
# inputs = np.array([[]])
# target = np.array([])
# #declaring the model type
# models = models.FeedForward()
# #add input layer
# models.add_dense(2, activation='Relu')
# #add hidden layer
# models.add_dense(10, activation='Relu')
# #add output layer
# models.add_dense(1, activation='softmax')
# #optinal but default is optimizer is gradient decent, default loss is mean square error
# models.model_optimizers(optimizer='gradient decent', loss='mean square error')
# #trains the model
# models.train_model(input_data=inputs, verify_data=target, batch_size=36, epochs=500)
# #saves the model
# models.save('file.dpr')
# # summarizes th data
# models.summery()
#
#
# # run an input through the model
# models.run(np.array([3.4]))
inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] + i[1]] for i in inputs])
models = models.FeedForward()
#add input layer
models.add_dense(2, activation='Relu')
#add hidden layer
models.add_dense(5, activation='Relu')
#add output layer
models.add_dense(1, activation='softmax')
#optinal but default is optimizer is gradient decent, default loss is mean square error
models.model_optimizers(optimizer='gradient decent', loss='mean square error')
#trains the model
models.train_model(input_data=inputs, verify_data=targets, batch_size=36, epochs=500)
