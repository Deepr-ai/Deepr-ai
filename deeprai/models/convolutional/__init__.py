class CNN:
    def __init__(self, *input_matrix, padding=0, strides=0, output_labels=None):
        # During assembly, make layer reshape
        self.output_labels = output_labels
        self.layers = [list(input_matrix)]
        self.kernels = []  # valid
        self.event_stack = []
        if padding != 0:
            self.padding_layer(padding)

    def pass_vals(self):
        pass
