import numpy as np
from activations import sigmoid, sigmoid_backward
from activations import relu, relu_backward
from weights_initializers import init_weights


class Dense:

    def __init__(self, input_shape, units, activation='relu', weights='glorot_uniform',
                 bias='zeros', weights_scale=0.05, bias_scale=0.05):

        self.batch_size = input_shape[0]
        self.input_units = input_shape[1]
        self.output_units = units
        self.output_shape = (self.batch_size, self.output_units)
        self.weights = init_weights(weights, shape=(self.input_units, self.output_units),
                                    scale=weights_scale)
        self.weights = np.array(self.weights, np.float32)
        self.bias = init_weights(bias, shape=(1, self.output_units), scale=bias_scale)
        self.prev_act = np.zeros((self.batch_size, self.output_units), dtype=np.float32)
        self.A = None
        if activation == 'relu':
            self.activation_fn = relu
            self.backward_activation_fn = relu_backward
        elif activation == 'sigmoid':
            self.activation_fn = sigmoid
            self.backward_activation_fn = sigmoid_backward
        else:
            raise Exception('Activation function not supported')

    def forward(self, prev_act):
        self.prev_act = prev_act.copy()
        self.A = self.activation_fn(np.dot(prev_act, self.weights) + self.bias)
        return self.A

    def backprop(self, dA, learning_rate = 0.01):
        dA = self.backward_activation_fn(dA, self.A)
        dW = np.dot(self.prev_act.T, dA) / self.batch_size
        db = dA.mean(axis=0, keepdims=True) / self.batch_size

        prev_dA = np.dot(dA, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return prev_dA
