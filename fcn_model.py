from neural_network import Sequential
from dense import Dense

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Creating Dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

X = StandardScaler().fit_transform(X)

# Splitting Dataset
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# Reshaping labels array into required shape -> (1, n_datapoints) for binary predicton
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

# Initializing Model
model = Sequential()

layer = Dense(input_shape=(None, 2), units=4)
output_shape = layer.output_shape
model.add(layer)

layer = Dense(input_shape=output_shape, units=6)
output_shape = layer.output_shape
model.add(layer)

layer = Dense(input_shape=output_shape, units=6)
output_shape = layer.output_shape
model.add(layer)

layer = Dense(input_shape=output_shape, units=4)
output_shape = layer.output_shape
model.add(layer)

layer = Dense(input_shape=output_shape, units=1, activation='sigmoid')
output_shape = layer.output_shape
model.add(layer)

model.compile()

# Training Model
model.fit(X_train, y_train, 10000, 16, X_test, y_test)

# Evaluating Model
vals = model.evaluate(X_test, y_test)
print(f"Test accuracy is {vals['accuracy']*100}")

# Plotting Test Set and predictions by model
y_pred = model.predict(X_test)

y_pred[y_pred > 0.5] = 1.
y_pred[y_pred <= 0.5] = 0
scat = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='*')
plt.savefig("training_set.png")
scat.remove()
plt.draw()
scat = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*')
plt.savefig("test_set.png")
scat.remove()
plt.draw()
scat = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='*')
plt.savefig("predictions.png")