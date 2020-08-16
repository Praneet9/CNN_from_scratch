import numpy as np


class Conv:

    def __init__(self, input_shape, n_filters=32, stride=1, kernel=3, padding=1):
        if kernel % 2 == 0:
            raise ValueError('kernel cannot be even')
        self.batch_size = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_channels = input_shape[3]
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weights = np.random.rand(kernel, kernel, input_shape[-1], n_filters)
        self.bias = np.random.rand(1, 1, 1, n_filters)
        self.n_filters = n_filters
        self.output_height = int(1 + ((self.input_height - kernel + 2 * padding) / stride))
        self.output_width = int(1 + ((self.input_width - kernel + 2 * padding) / stride))
        self.output_shape = (self.batch_size, self.output_height, self.output_width, self.n_filters)
        self.prev_act = np.zeros(input_shape)

    def convolution(self, receptive_field, W, b):
        W = np.rot90(W, 2)
        Z = np.multiply(receptive_field, W)
        Z = np.sum(Z)
        Z = Z + float(b)
        return Z

    def add_padding(self, batch):
        padded_batch = np.pad(batch, ((0, 0),
                           (self.padding, self.padding),
                           (self.padding, self.padding),
                           (0, 0)),
                   'constant')
        return padded_batch

    def forward(self, prev_act):

        Z = np.zeros(self.output_shape)
        padded_batch = self.add_padding(prev_act)
        for m in range(self.batch_size):
            image_padded = padded_batch[m]
            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.n_filters):
                        vertical_start = h*self.stride
                        horizontal_start = w*self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel
                        receptive_field = image_padded[vertical_start:vertical_end, horizontal_start:horizontal_end]
                        Z[m, h, w, c] = self.convolution(receptive_field, self.weights[:, :, :, c], self.bias[:, :, :, c])

        self.prev_act = prev_act.copy()
        return Z

    def backprop(self, dZ, learning_rate = 0.01):

        prev_dA = np.zeros((self.batch_size, self.input_height, self.input_width, self.input_channels))

        dW = np.zeros((self.kernel, self.kernel, self.input_channels, self.n_filters))
        db = np.zeros((1, 1, 1, self.n_filters))

        prev_act_padded = self.add_padding(self.prev_act)
        prev_dA_padded = self.add_padding(prev_dA)

        for m in range(self.batch_size):
            prev_act_pad = prev_act_padded[m]
            prev_dA_pad = prev_dA_padded[m]

            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.n_filters):
                        vertical_start = h * self.stride
                        horizontal_start = w * self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel

                        receptive_field = prev_act_pad[vertical_start:vertical_end, horizontal_start:horizontal_end]
                        prev_dA_pad[vertical_start:vertical_end,
                                    horizontal_start:horizontal_end, :] += self.weights[:, :, :, c] * dZ[m, h, w, c]
                        dW[:, :, :, c] += receptive_field * dZ[m, h, w, c]
                        db[:, :, :, c] += dZ[m, h, w, c]

            prev_dA[m, :, :, :] = prev_dA_pad if self.padding == 0 else prev_dA_pad[self.padding:-self.padding,
                                                                                    self.padding:-self.padding, :]
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return prev_dA
