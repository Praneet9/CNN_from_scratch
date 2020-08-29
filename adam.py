import numpy as np


class Adam:

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.learning_rate = learning_rate
        self.v = []
        self.s = []

    def init_params(self, layers):

        for layer in layers:
            if hasattr(layer, 'weights'):
                w_b = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }

                self.v.append(w_b)
                self.s.append(w_b.copy())
            else:
                w_b = {
                    'weights': None,
                    'bias': None
                }

                self.v.append(w_b)
                self.s.append(w_b.copy())

    def optimize(self, layers, gradients, batch_size, iteration):

        for idx, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                continue

            # weights optimization
            g = gradients[idx][0] / batch_size

            self.v[idx]['weights'] = self.beta1 * self.v[idx]['weights'] + (1. - self.beta1) * g
            self.s[idx]['weights'] = self.beta2 * self.s[idx]['weights'] * (1. - self.beta2) * np.square(g)

            v_bias_corr = self.v[idx]['weights'] / (1. - self.beta1 ** iteration)
            sqr_bias_corr = self.s[idx]['weights'] / (1. - self.beta2 ** iteration)

            adjustment = self.learning_rate * v_bias_corr / (np.sqrt(sqr_bias_corr) + self.eps)
            layer.weights -= adjustment

            # bias optimization
            g = gradients[idx][1] / batch_size

            self.v[idx]['bias'] = self.beta1 * self.v[idx]['bias'] + (1. - self.beta1) * g
            self.s[idx]['bias'] = self.beta2 * self.s[idx]['bias'] * (1. - self.beta2) * np.square(g)

            v_bias_corr = self.v[idx]['bias'] / (1. - self.beta1 ** iteration)
            sqr_bias_corr = self.s[idx]['bias'] / (1. - self.beta2 ** iteration)

            adjustment = self.learning_rate * v_bias_corr / (np.sqrt(sqr_bias_corr) + self.eps)
            layer.bias -= adjustment
