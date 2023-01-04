from deeprai import models
"""
Final syntax structure
"""


model = models.Convolutional()
model.add_kernel(2, [3,2], actation='relu', dim='2d')
model.add_pool([3,3], mode = 'max', dim='2d')
model.flatten()
model.add_dense(23, actavation='tanh')
model.add_dense(12, actavation='softmax')
model.model_optimizers(optimizer='adem', loss='ccs')
model.train_model(x,y, verify_data=(x, y), batch_size=20)
model.save('m1.dpr')
model.summery()

"""
Turbo
"""
from deeprai.tools import Turbo as dpr

model = dpr.build(model="conv", autofit = True)
model.set_data('file.csv')
model.run(output_file='path/to/file')

