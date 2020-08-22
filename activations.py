import numpy as np


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dZ, Z):
    sig = Z
    return dZ * sig * (1 - sig)

def relu_backward(dZ, Z):
    return dZ * np.heaviside(Z, 0)