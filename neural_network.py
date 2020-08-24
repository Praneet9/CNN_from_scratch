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

        # TODO - validations

        for epoch in range(1, epochs):
            idx = 0
            batch_no = 0

            if y.shape[1] % batch_size:
                no_batches = (y.shape[1] // batch_size) + 1
            else:
                no_batches = y.shape[1] // batch_size

            print(f"\nEpoch {idx+1}/{epochs}")
            loss = []
            accuracies = []
            while idx <= y.shape[1]:
                batch_no += 1

                x_batch = x[idx:idx+self.batch_size]
                y_batch = y[:, idx:idx+self.batch_size]

                idx += self.batch_size
                prev_act = x_batch.copy()
                for layer in self.layers:
                    prev_act = layer.forward(prev_act)

                cost, dZ = self.loss_function(y_batch, prev_act)

                loss.append(cost)
                y_pred = prev_act.reshape(1, -1)
                accuracy = np.mean(np.round(y_pred) == y_batch)
                accuracies.append(accuracy)

                print(f"Batch {batch_no}/{no_batches} Loss: {sum(loss)/len(loss)} "
                      f"Accuracy: {sum(accuracies)/len(accuracies)}", end='\r')
                for layer in reversed(self.layers):
                    dZ = layer.backprop(dZ)

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