import numpy as np


class Conv():

    def __init__(self, input_shape, n_filters=32, stride=1, kernel=3, padding=1):
        if kernel % 2 == 0:
            raise ValueError('kernel cannot be even')
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.kernel_array = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        self.n_filters = n_filters
        filters = np.zeros((input_shape[0] - kernel + 1,
                            input_shape[0] - kernel + 1,
                            self.n_filters))
        self.output = np.zeros((input_shape))

    def convolution(self, receptive_field, W, b):
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

    def forward(self, prev_act, W, b):
        batch_size = prev_act.shape[0]
        height = int(1 + ((prev_act.shape[1] - self.kernel + 2 * self.padding)/self.stride))
        width = int(1 + ((prev_act.shape[2] - self.kernel + 2 * self.padding)/self.stride))

        Z = np.zeros((batch_size, height, width, self.n_filters))
        padded_batch = self.add_padding(prev_act)
        for m in batch_size:
            image_padded = padded_batch[m]
            for h in range(height):
                for w in range(width):
                    for c in range(self.n_filters):
                        vertical_start = h*self.stride
                        horizontal_start = w*self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel
                        # Why not channel?
                        receptive_field = image_padded[vertical_start:vertical_end, horizontal_start:horizontal_end]
                        Z[m, h, w, c] = self.convolution(receptive_field, W[:, :, :, c], b[:, :, :, c])

        weights = (prev_act, W, b)
        return Z, weights

    def backprop(self, weights, dZ):
        prev_act, W, b = weights

        _, prev_height, prev_width, prev_channels = prev_act.shape

        batch_size, height, width, channels = dZ.shape
        prev_dA = np.zeros((batch_size, prev_height, prev_width, prev_channels))
        dW = np.zeros((self.kernel, self.kernel, prev_channels, channels))
        db = np.zeros((1, 1, 1, channels))

        prev_act_padded = self.add_padding(prev_act)
        prev_dA_padded = self.add_padding(prev_dA)

        for m in range(batch_size):
            prev_act_pad = prev_act_padded[m]
            prev_dA_pad = prev_dA_padded[m]

            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        vertical_start = h * self.stride
                        horizontal_start = w * self.stride
                        vertical_end = vertical_start + self.kernel
                        horizontal_end = horizontal_start + self.kernel

                        receptive_field = prev_act_pad[vertical_start:vertical_end,
                                          horizontal_start:horizontal_end] # Why not channel?
                        prev_dA_pad[vertical_start:vertical_end,
                                    horizontal_start:horizontal_end, :] += W[:, :, :, c] * dZ[m, h, w, c]
                        dW[:, :, :, c] += receptive_field * dZ[m, h, w, c]
                        db[:, :, :, c] += dZ[m, h, w, c]

            prev_dA[m, :, :, :] = prev_dA_pad if self.padding == 0 else prev_dA_pad[self.padding:-self.padding,
                                                                                    self.padding:-self.padding, :]

        return prev_dA, dW, db
