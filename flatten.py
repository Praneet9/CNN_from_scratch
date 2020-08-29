import numpy as np


class Flatten:

    def __init__(self, input_shape):

        self.input_height = input_shape[-3]
        self.input_width = input_shape[-2]
        self.input_channels = input_shape[-1]
        self.output_shape = (None, self.input_height * self.input_width * self.input_channels)

    def forward(self, prev_act):
        batch_size = prev_act.shape[0]

        prev_act = np.array(prev_act)
        activation = prev_act.reshape(batch_size, -1)

        return activation

    def backprop(self, dA):
        batch_size = dA.shape[0]

        prev_dA = dA.reshape((batch_size, self.input_height, self.input_width, self.input_channels))

        return prev_dA, [None, None]