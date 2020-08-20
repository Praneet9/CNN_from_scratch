import numpy as np

def binary_cross_entropy(Y, P_hat):
    """
    This function computes Binary Cross-Entropy(bce) Cost and returns the Cost and its
    derivative.
    This function uses the following Binary Cross-Entropy Cost defined as:
    => (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    Args:
        Y: labels of data
        P_hat: Estimated output probabilities from the last layer, the output layer
    Returns:
        cost: The Binary Cross-Entropy Cost result
        dP_hat: gradient of Cost w.r.t P_hat
    """
    m = Y.shape[1]  # m -> number of examples in the batch

    cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    cost = np.squeeze(cost)

    dP_hat = (1/m) * (-(Y/P_hat) + ((1-Y)/(1-P_hat)))

    return cost, dP_hat