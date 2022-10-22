from deeprai import models
model = models.Convolutional(input_data)
model.add_kernel(2, [3,2], actation='relu', dim='2d')
model.add_pool([3,3], mode = 'max', dim='2d')
model.flatten()
model.add_dense(23, actavation='tanh')
model.add_dense(12, actavation='softmax')
model.run()
model.summery()
