import numpy as np


class Dense:

    def __init__(self, input_shape, units):

        self.batch_size = input_shape[0]
        self.input_units = input_shape[1]
        self.output_units = units
        self.output_shape = (self.batch_size, self.output_units)
        self.weights = np.random.rand(self.input_units, self.output_units)
        self.bias = np.random.rand(1, self.output_units)
        self.prev_act = np.zeros((self.batch_size, self.output_units))

    def forward(self, prev_act):
        self.prev_act = prev_act.copy()
        return np.dot(prev_act, self.weights) + self.bias

    def backprop(self, dZ, learning_rate = 0.01):
        dW = np.dot(self.prev_act.T, dZ)
        db = dZ.mean(axis=0, keepdims=True)

        prev_dA = np.dot(dZ, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return prev_dA
