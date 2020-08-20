import numpy as np
from loss import binary_cross_entropy


class Sequential:

    def __init__(self):

        self.loss_function = None
        self.epochs = None
        self.layers = []
        self.batch_size = 4

    def add(self, layer):

        self.layers.append(layer)

    def compile(self, loss='binary_cross_entropy'):

        if loss == 'binary_cross_entropy':
            self.loss_function = binary_cross_entropy
        else:
            raise ValueError(f"{loss} has not been added yet!")

        for idx, layer in enumerate(self.layers):
            print(f"Layer {str(idx+1)} ; Name: {layer.__class__.__name__} ; Output Shape: {layer.output_shape}")

    def fit(self, x, y, epochs, batch_size, val_x, val_y):
        self.epochs = epochs
        self.batch_size = batch_size

        # iterate for number of epochs
        # take steps of n_samples / batch_size
        # iterate forward through every layer
        # compute cost
        # print loss and accuracy
        # iterate backward through every layer
        # update all weights


    def predict(self, x):

        # take steps of n_samples / batch_size
        # iterate forward through every layer
        # return probabilities
        pass

    def evaluate(self, x, y):

        # take steps of n_samples / batch_size
        # iterate forward through every layer
        # update accuracy at every step
        # return accuracy
        pass