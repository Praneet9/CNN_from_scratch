import numpy as np


def init_weights(init_type, shape, scale=0.05):
    if init_type == 'uniform':
        return uniform(shape, scale)
    elif init_type == 'normal':
        return normal(shape, scale)
    elif init_type == 'glorot_uniform':
        return glorot_uniform(shape)
    elif init_type == 'zeros':
        return zeros(shape)
    else:
        ValueError("Invalid initializer")


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.05):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)


def zeros(shape):
    return np.zeros(shape)