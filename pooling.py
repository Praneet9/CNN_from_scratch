import numpy as np


class Pooling():

    def __init__(self, pooling_type='max', stride=2, kernel=3):

        self.stride = stride
        self.pooling_type = pooling_type
        self.kernel = kernel

    def forward(self, prev_act):

        batch_size = prev_act.shape[0]
        height = int(1 + (prev_act.shape[1] - self.kernel) / self.stride)
        width = int(1 + (prev_act.shape[2] - self.kernel) / self.stride)
        channels = prev_act.shape[3]

        activation = np.zeros((batch_size, height, width, channels))

        for m in batch_size:
            image = prev_act[m]
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        vertical_start = h*self.stride
                        horizontal_start = w*self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel

                        receptive_field = image[vertical_start:vertical_end, horizontal_start:horizontal_end, c]
                        if self.pooling_type == 'max':
                            activation[m, h, w, c] = np.max(receptive_field)
                        elif self.pooling_type == 'avg':
                            activation[m, h, w, c] = np.mean(receptive_field)

        return activation, prev_act

    def backprop(self, dA, prev_act):

        batch_size, height, width, channels = dA.shape
        prev_dA = np.zeros(prev_act.shape)

        for m in range(batch_size):
            image = prev_act[m]
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
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

        return prev_dA