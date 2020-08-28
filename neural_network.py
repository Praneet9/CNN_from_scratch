import numpy as np
from loss import binary_cross_entropy


class Sequential:

    def __init__(self):

        self.loss_function = None
        self.epochs = None
        self.layers = []
        self.batch_size = 4
        self.loss = []
        self.accuracies = []
        self.val_loss = []
        self.val_accuracies = []

    def add(self, layer):

        self.layers.append(layer)

    def compile(self, loss='binary_cross_entropy'):

        if loss == 'binary_cross_entropy':
            self.loss_function = binary_cross_entropy
        else:
            raise ValueError(f"{loss} has not been added yet!")

        for idx, layer in enumerate(self.layers):
            print(f"Layer {str(idx+1)} ; Name: {layer.__class__.__name__} ; Output Shape: {layer.output_shape}")

    def fit(self, x, y, epochs, batch_size, val_x=None, val_y=None):
        self.epochs = epochs
        self.batch_size = batch_size

        for epoch in range(1, epochs):
            idx = 0
            batch_no = 0

            if y.shape[1] % batch_size:
                no_batches = (y.shape[1] // batch_size) + 1
            else:
                no_batches = y.shape[1] // batch_size

            print(f"\nEpoch {epoch+1}/{epochs}")

            while idx <= y.shape[1]:
                batch_no += 1

                x_batch = x[idx:idx+self.batch_size]
                y_batch = y[:, idx:idx+self.batch_size]

                idx += self.batch_size
                prev_act = x_batch.copy()
                prev_act = self.predict(prev_act)

                cost, dZ = self.loss_function(y_batch, prev_act)

                self.loss.append(cost)
                y_pred = prev_act.reshape(1, -1)
                accuracy = np.mean(np.round(y_pred) == y_batch)
                self.accuracies.append(accuracy)

                print(f"Batch {batch_no}/{no_batches} loss: {sum(self.loss)/len(self.loss)} "
                      f"accuracy: {sum(self.accuracies)/len(self.accuracies)}", end='\r')

                for layer in reversed(self.layers):
                    dZ = layer.backprop(dZ)

            print(f"Batch {batch_no}/{no_batches} loss: {sum(self.loss) / len(self.loss)} "
                  f"accuracy: {sum(self.accuracies) / len(self.accuracies)}", end='')

            if val_x is not None:
                val_dict = self.evaluate(val_x, val_y)
                self.val_loss.extend(val_dict['batch_loss'])
                self.val_accuracies.extend(val_dict['batch_accuracy'])

                print(f" val_loss: {sum(self.val_loss)/len(self.val_loss)} "
                      f"val_accuracy: {sum(self.val_accuracies)/len(self.val_accuracies)}")

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def evaluate(self, x, y):

        idx = 0
        batch_no = 0
        loss, accuracies = [], []
        while idx <= y.shape[1]:
            batch_no += 1

            x_batch = x[idx:idx + self.batch_size]
            y_batch = y[:, idx:idx + self.batch_size]

            idx += self.batch_size
            prev_act = x_batch.copy()
            prev_act = self.predict(prev_act)

            cost, _ = self.loss_function(y_batch, prev_act)

            loss.append(cost)
            y_pred = prev_act.reshape(1, -1)
            accuracy = np.mean(np.round(y_pred) == y_batch)
            accuracies.append(accuracy)

        return {
            'loss': sum(loss) / len(loss),
            'accuracy': sum(accuracies) / len(accuracies),
            'batch_loss': loss,
            'batch_accuracy': accuracies
        }
