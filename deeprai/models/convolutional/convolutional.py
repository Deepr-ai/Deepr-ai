class Convolutional:
    def __init__(self):
        self.layers = []
        self.kernels = []
        self.bias = []
        self.padding = None
        self.strides = None
        self.output_labels = None

    def add_kernel(self, amount, shape, max_size: int = 2, ):
        self.stack_layer.create_kernel(amount, shape, max_size)

    def add_pool(self, shape):
        self.stack_layer.create_pool()

    def add_dense(self, neurons, activation=''):
        self.stack_layer.create_dense()

    def flatten(self):
        self.stack_layer.create_flat()

    def merge(self, merge_type='sum'):
        self.stack_layer.create_merge()

    def model_optimizers(self, optimizer='', loss=''):
        pass

    def train_model(self, input_data, verify_data, batch_size=10, epoches=500, learning_rate=0.1, momentum=0.6, verbose=True):
        pass

    def summery(self):
        pass