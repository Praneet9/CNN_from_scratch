import numpy as np


class Flatten:

    def __init__(self, input_shape):

        self.batch_size = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_channels = input_shape[3]
        self.output_shape = (self.batch_size, self.input_height * self.input_width * self.input_channels)

    def forward(self, prev_act):
        prev_act = np.array(prev_act)
        activation = prev_act.reshape(self.batch_size, -1)

        return activation

    def backprop(self, dA):
        prev_dA = dA.reshape((self.batch_size, self.input_height, self.input_width, self.input_channels))

        return prev_dA