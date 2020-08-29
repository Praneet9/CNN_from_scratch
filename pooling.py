import numpy as np


class Pooling:

    def __init__(self, input_shape, pooling_type='max', stride=2, kernel=2):

        self.input_height = input_shape[-3]
        self.input_width = input_shape[-2]
        self.input_channels = input_shape[-1]
        self.output_channels = input_shape[-1]
        self.stride = stride
        self.pooling_type = pooling_type
        self.output_height = int(1 + ((self.input_height - kernel) / stride))
        self.output_width = int(1 + ((self.input_width - kernel) / stride))
        self.kernel = kernel
        self.output_shape = (None, self.output_height, self.output_width, self.output_channels)
        self.prev_act = None

    def forward(self, prev_act):
        batch_size = prev_act.shape[0]
        activation = np.zeros((batch_size, self.output_height, self.output_width, self.output_channels))

        for m in range(batch_size):
            image = prev_act[m]
            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.output_channels):
                        vertical_start = h*self.stride
                        horizontal_start = w*self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel

                        receptive_field = image[vertical_start:vertical_end, horizontal_start:horizontal_end, c]
                        if self.pooling_type == 'max':
                            activation[m, h, w, c] = np.max(receptive_field)
                        elif self.pooling_type == 'avg':
                            activation[m, h, w, c] = np.mean(receptive_field)

        self.prev_act = prev_act.copy()
        return activation

    def backprop(self, dA):
        batch_size = dA.shape[0]
        prev_dA = np.zeros(self.prev_act.shape)

        for m in range(batch_size):
            image = self.prev_act[m]
            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.output_channels):
                        vertical_start = h * self.stride
                        horizontal_start = w * self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel

                        if self.pooling_type == "max":
                            receptive_field = image[vertical_start:vertical_end, horizontal_start:horizontal_end, c]
                            mask = (receptive_field == np.max(receptive_field))
                            prev_dA[m, vertical_start:vertical_end,
                                    horizontal_start:horizontal_end, c] += np.multiply(mask, dA[m, h, w, c])
                        elif self.pooling_type == "avg":
                            average = dA[m, h, w, c]
                            shape = (self.kernel, self.kernel)
                            average = average / (shape[0] * shape[1])
                            average = np.ones(shape) * average
                            prev_dA[m, vertical_start:vertical_end, horizontal_start:horizontal_end, c] += average

        return prev_dA, [None, None]
